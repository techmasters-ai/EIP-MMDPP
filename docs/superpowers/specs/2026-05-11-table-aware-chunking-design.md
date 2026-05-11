# Table-Aware Chunking — Design

**Status:** Draft 2026-05-11 (rev. 4 after second review pass)
**Branch:** `feat/table-aware-chunking`
**Related:**
- `2026-05-05-section-aware-table-fact-synthesis-design.md` — prior spec for `_table_facts.py`, which was **built, then reverted from production** on 2026-05-06 (see §1 below). Modules remain on disk; not currently called by `run_extraction_pass`.
- `2026-05-06-table-identity-rewrite-and-field-overlay-design.md` — `table_overlay.py` Phase 0/0.5 wiring. **Untouched** by this spec.

---

## 1. Problem

### 1.1 What runs in production today

Reading `docker/docling-graph/app/main.py:566-574`:

> *"Section-aware table-fact synthesis (table_facts.py + alias_map.py) was built and validated in the 2026-05-06 plan, **then reverted here** after cross-pass measurement showed the cost (+10-30% wall on docs with variants tables, +output truncation pressure) outweighed the benefit (+2 ✓ exact on airframe for 1 of 21 corpus docs; no improvement on kinematics/speed_timing; **propulsion fix landed but unverified**). Modules remain on disk in app/_table_facts.py + app/_alias_map.py with full tests; re-enable when the corpus has more variants-table documents to amortize the maintenance cost. See TODO #84."*

So the **current production behavior** for tables is:

- **Graph extraction side:** raw flattened table text appears in `docling_document_json["texts"]` for each table (Docling-generated). `extract_table_overlay()` (which lives inside `_table_facts.py` but is a separate entry point from `synthesize_table_facts`) runs at `main.py:1185` on every pass to produce the Phase 0/0.5 overlay (consumed at merge time by `app/services/table_overlay.py`) — **this is production-live, not sidelined**. Only `synthesize_table_facts()` was reverted; it has no production caller and is only invoked by tests. `run_extraction_pass` consumes raw `texts[]` as input.
- **Embedding side (primary path):** `app/workers/pipeline.py:5500-5634` — `HybridChunker` (the Docling native chunker) chunks the enriched `DoclingDocument` from `docling_document.json`. Tables are emitted as native chunks whose `meta.doc_items` reference `#/tables/N` self_refs. The chunker uses `BAAI/bge-m3` as tokenizer (`pipeline.py:5524-5525`) with `max_tokens=settings.embedding_chunk_max_tokens=512`. **This is the path that runs for enriched docs (i.e., almost all production docs).**
- **Embedding side (legacy fallback):** `app/services/chunking.py:113-126`'s `structure_aware_chunk` emits one opaque `modality="table"` chunk per table element. Only runs when `enrichments.version` is `None` or HybridChunker raises an exception. Not the primary production path.

The embedding model is `BAAI/bge-large-en-v1.5` (hard limit 512 tokens). HybridChunker uses bge-m3 tokenizer with budget 512 to chunk; the produced chunks are then embedded by bge-large-en-v1.5. The two tokenizers can disagree on the same string by ±5%; this is an existing pipeline characteristic that this spec inherits and does not attempt to fix.

### 1.2 What's wrong with that

For wide spec-sheet tables like the SA-2 variants table (~2,000–3,000 tokens of flattened content):

- **Embedding side (HybridChunker):** HybridChunker emits a table either as one whole chunk (when it fits within 512 bge-m3 tokens) or splits it at table boundaries Docling provides. Splitting a wide variants table at arbitrary row boundaries fragments variant context — e.g., `S-75M2`'s identity row ends up in one chunk and its `Max Range` row in another. Variant-specific queries lose precision because no single chunk holds entity + property together.
- **Embedding side (legacy fallback):** when the legacy path runs, the chunk is silently truncated to the first 512 tokens of flattened content.
- **Graph extraction side:** the raw flattened table is the "active source of column-arithmetic confusion" the prior `_table_facts.py` work attempted to address. That work documented a systematic off-by-one row→field shift on `missile_propulsion`, but the propulsion fix was reverted before being empirically verified.

### 1.3 What this spec does

A table-aware chunking layer that:
1. Normalizes Docling table cells into a renderer-agnostic model.
2. Emits structured per-entity-column chunks on the graph side (replacing the raw flattened table text).
3. Emits a graduated multi-view set of chunks on the embedding side (always a summary; whole-table chunk if under budget; per-column otherwise).
4. Preserves cell-level provenance through both sides.

The design is informed by an external analysis (provided 2026-05-11). It adapts that analysis to the existing pipeline. The dormant `_table_facts.py` + `_alias_map.py` modules are preserved as an A/B experimental path — **not** as a "validated baseline" (no such baseline exists; see §1.1).

## 2. Goals

1. **Embedding-side retrieval precision.** Variant-specific queries reliably retrieve the correct per-variant chunk for wide tables.
2. **Graph-side self-contained chunks.** Replace the raw flattened table text with per-entity-column chunks that preserve identity + section + spec rows together.
3. **Provenance traceability.** Every emitted chunk carries `cell_refs` pointing back to `#/tables/N/data/table_cells/M` entries.
4. **No regression vs today's behavior.** Extraction quality on the SA-2 corpus (and other tested corpora) is *at least as good* as the current raw-flattened-table production pipeline, measured against a baseline captured immediately before any code lands. This is a **merge gate**, not a post-merge measurement (§15, §19).
5. **Master kill-switch correctness.** When both `*_NORMALIZATION_ENABLED` flags are false, code paths are *byte-identical* to today's behavior (raw blob, no synthesis, no normalization).
6. **Experimental path preserved.** The reverted `_table_facts.py` + `_alias_map.py` codepath stays on disk and is reachable behind a separate flag — for A/B experimentation, not as a rollback target.

## 3. Non-Goals (explicit out-of-scope)

- Multilingual row labels.
- Prose-table hybrids (tables embedded in flowing prose without proper Docling parsing).
- Cross-table consolidation.
- Notebook outcome-tracker `facts/pass` column updates.
- Retrieval-side reranking boosts by `chunk_kind` (enabled by the new column, but a separate experiment).
- UI surfacing of the `table_chunk` block in the document viewer.
- Generated `§12b` prose from normalized tables.
- A backward-compatibility re-chunk script — rollout uses `./manage.sh --blow-away` + re-ingest per user decision.
- Auto-extending the section-keyword list or the spec-row keyword set — these stay hand-coded; additions are one-line changes + a test.

## 4. Architecture Overview

```
docling_document_json["tables"]
            │
            ▼
   ┌────────────────────┐
   │  normalize_tables  │   ← single pass; renderer-agnostic
   └────────┬───────────┘
            │
            ▼
   list[NormalizedTable]
            │
   ┌────────┴───────────┐
   ▼                    ▼
render_for_graph    render_for_embedding
   │                    │
   ▼                    ▼
GraphTableChunk[]   EmbeddingTableChunk[]
   │                    │
   ▼                    ▼
docling-graph        app/services/chunking.py
texts[] (LLM input)  → TextChunk rows + ArcadeDB vertices
```

The `NormalizedTable` model is the only contract between normalization and rendering. Both renderers are pure functions sharing one internal helper (`_render_column_as_text`) that produces the identity+sections+rows block — one source of truth for chunk format.

## 5. Module Layout

```
app/services/table_normalization/
├── __init__.py                 # public API: normalize_tables(), render_for_graph(), render_for_embedding(), is_normalization_enabled_*()
├── models.py                   # NormalizedTable, NormalizedRow, NormalizedColumn, NormalizedCell, TableSection, ChunkKind
├── detect.py                   # shape detection
├── normalize.py                # docling table_cells → NormalizedTable
├── render_graph.py             # NormalizedTable → list[GraphTableChunk]
├── render_embedding.py         # NormalizedTable → list[EmbeddingTableChunk]
├── tokens.py                   # bge-m3 tokenizer-aware size check
└── config.py                   # env-flag reading + default thresholds
```

**Cross-image distribution.** `app/services/table_normalization/` is the single source of truth. `docker/docling-graph/Dockerfile` adds a `COPY app/services/table_normalization /app/app/services/table_normalization` directive at build time. `api`/`worker`/`worker-graph` use the bind-mounted `./app` and pick up changes without rebuild. Docling-graph requires `docker compose build docling-graph` to pick up changes (per `feedback_container_rebuild`).

**Modules deleted by this work:**
- `docker/docling-graph/app/_table_pivot.py` — already deprecated. Only consumer is `docker/docling-graph/tests/test_table_pivot.py` (imports via `importlib.util` as deprecation scaffolding). Both files deleted. Grep verified: `grep -rn "_table_pivot" --include="*.py" .` returns only these two paths.
- `docker/docling-graph/tests/test_table_pivot.py` — deleted alongside its target module.

**Modules preserved unchanged:**
- `docker/docling-graph/app/_table_facts.py` — **production-live** for `extract_table_overlay()` (`main.py:1185`). Untouched. `synthesize_table_facts()` (its other entry point) is the reverted experimental path; gated by the new `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` flag (default `false`).
- `docker/docling-graph/app/_alias_map.py` — used by `synthesize_table_facts()` only. Stays on disk; not in production today. Reachable only via the experimental flag.
- `docker/docling-graph/tests/test_table_facts_*.py` — stay green; they test the module's functions (both `extract_table_overlay` and `synthesize_table_facts`) directly via module imports.
- `app/services/table_overlay.py` — untouched. Phase 0/0.5 merge-time machinery consumes the overlay produced by `extract_table_overlay()` (which reads `body.docling_document_json["tables"]` directly, not `texts[]`). The overlay machinery does not depend on flat `texts[]` representation of tables. Verified at `main.py:1185-1228`.
- `app/services/chunking.py` (legacy `structure_aware_chunk`) — modified per §10.2 with a master kill-switch for the legacy embedding path.

## 6. Data Model (`models.py`)

```python
class Shape(str, Enum):
    COLUMN_MAJOR = "column_major"
    ROW_MAJOR    = "row_major"
    HYBRID       = "hybrid"          # column-major + multiple identity rows
    OTHER        = "other"           # skip; fall back to raw rendering


class ChunkKind(str, Enum):
    TABLE_SUMMARY        = "table_summary"
    TABLE_WHOLE          = "table_whole"
    TABLE_ENTITY_COLUMN  = "table_entity_column"
    TABLE_ENTITY_SECTION = "table_entity_section"


@dataclass(frozen=True)
class CellRef:
    table_index: int                 # index into doc_json["tables"]
    row_idx: int
    col_idx: int
    self_ref: str                    # "#/tables/3/data/table_cells/42"


@dataclass(frozen=True)
class NormalizedCell:
    row_idx: int
    col_idx: int
    row_label: str | None
    column_identity: dict[str, str]  # full identity dict for this column
    section: str | None
    value: str                       # raw text; no numeric coercion
    unit: str | None                 # inherited from row label
    cell_ref: CellRef


@dataclass(frozen=True)
class NormalizedRow:
    row_idx: int
    label: str
    is_identity_row: bool
    is_section_header: bool
    section: str | None
    unit: str | None                 # extracted from label suffix


@dataclass(frozen=True)
class NormalizedColumn:
    col_idx: int
    identity: dict[str, str]
    display_name: str                # heuristic: Industry → Military → NATO → Missile Type; fallback "col-{n}"


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
    raw_markdown: str                # captured for OTHER fallback + small-table whole rendering. Source resolution rule: §8 step 7.
```

All `frozen=True`. Immutable post-construction. Row-major tables represented identically: `NormalizedColumn` corresponds to a *row* in the source, with its row-header text as the identity.

**On the `column_index` / `entity_display_name` fields on chunk types (§9, §10):** these duplicate information derivable from `cell_refs`. They are **denormalized** on purpose — kept on the chunk so the retrieval response (`/v1/retrieval/query`) and graph-extraction provenance can render entity attribution without a second lookup. Drift risk mitigated by the chunk being immutable and produced in one pass.

## 7. Shape Detection (`detect.py`)

Pure function: `detect_shape(table_cells, table_data) -> Shape`. Operates only on Docling-provided signals (`row_header`, `column_header`, `start_row_offset_idx`, `end_row_offset_idx`, `start_col_offset_idx`, `end_col_offset_idx`, `text`).

**Detection rules, in order:**

1. **Floor:** `num_rows < 4` or `num_cols < 4` → `OTHER`.

2. **`COLUMN_MAJOR` test:** ≥50% of non-empty `start_col_offset_idx == 0` cells have `row_header: True`, AND at least one row label matches a spec-row keyword (see closed list below).

3. **`HYBRID` upgrade:** Applied after `COLUMN_MAJOR`. Count rows 0..N where every data cell is identity-shaped (short text < 40 chars, non-numeric, non-blank). If ≥2 such rows exist at the *top* → upgrade to `HYBRID`.

4. **`ROW_MAJOR` test:** ≥50% of non-empty row-0 cells have `column_header: True`, AND row-0 contains spec keywords.

5. **Section-header detection (all non-OTHER shapes):** A row is a section header when all non-empty cells span the full data-column width (`col_span ≥ num_data_cols`) AND cell text matches a section keyword (closed list below).

6. **Fallback:** Anything not matching → `OTHER`. `normalize_tables` emits `NormalizedTable(shape=OTHER, cells=(), raw_markdown=...)`.

**Closed keyword lists (committed in `detect.py` — no `etc.` placeholders):**

```python
SPEC_ROW_KEYWORDS = frozenset({
    "max range", "min range", "range", "max altitude", "min altitude", "altitude",
    "max speed", "min speed", "speed", "velocity", "vmax", "vmin",
    "weight", "mass", "total weight", "warhead weight",
    "length", "width", "diameter", "span", "height",
    "max alt", "min alt",
    "missile type", "missile variant",
    "frequency", "wavelength", "power",
    "thrust", "burn time", "stage",
})

SECTION_KEYWORDS = frozenset({
    "missile", "1st stage", "2nd stage", "first stage", "second stage",
    "booster", "sustainer", "propulsion",
    "radar", "launcher", "guidance",
    "warhead", "fuze",
    "system performance", "performance",
})

IDENTITY_LABEL_KEYWORDS = frozenset({
    "designation", "variant", "type", "name",
    "industry designation", "military designation", "nato designation",
    "fan song variant", "radar variant",
    "system name", "system designation",
})
```

Adding a new keyword is a single-line change + a test fixture. **Diagnostics emit a WARNING when a table is classified `OTHER` despite having ≥4 rows AND ≥4 cols** (the "interesting-but-unrecognized" case), surfacing the row labels and column headers that failed to match. This makes keyword-list misses visible operationally (per `feedback_softfails_to_todo`).

**Diagnostics emitted alongside Shape:**
```python
{
    "shape": "column_major",
    "rows": 23,
    "cols": 12,
    "identity_rows": 4,
    "section_headers_detected": 2,
    "row_label_coverage_pct": 87,
    "spec_keyword_hits": 9,
    "fell_back_to_other": False,
}
```

## 8. Normalization (`normalize.py`)

Entry point:
```python
def normalize_tables(doc_json: dict) -> list[NormalizedTable]:
    """Pure function. Reads doc_json['tables']; never writes doc_json.
    Returns one entry per table including OTHER (with empty cells, raw_markdown set)."""
```

**Per-table pipeline:**

1. **Shape detection** via `detect.py`. `OTHER` → emit minimal `NormalizedTable`, skip steps 2–7.

2. **Build `NormalizedRow` list.**
   - Column-major/hybrid: iterate rows; row label = text of cell at `start_col_offset_idx == 0` (or merged label cell spanning col 0). Unit extracted from label suffix via regex `r"\(\s*([a-zA-Z/°²³]+)\s*\)\s*$"`.
   - Row-major: conceptually transposed; "row label" comes from column header in row 0; same logic.

3. **Build `NormalizedColumn` list (entities).**
   - Column-major/hybrid: each non-label column is one entity. Identity dict built by walking every identity row and reading the cell at that column. Empty identity cells fall back via `colspan` propagation.
   - `display_name` heuristic: prefer `Industry Designation`, then `Military Designation`, then `NATO Designation`, then `Missile Type`. If none → `"col-{col_idx}"` stable fallback.
   - Row-major: each non-header row becomes a `NormalizedColumn`.

4. **Section assignment.** Walk row list top-to-bottom; section-header rows reset the section context for all subsequent spec rows. Rows above the first section header have `section = None`.

5. **Build `NormalizedCell` list.** One cell per (spec row × entity column) pair. `value` = raw text (strip whitespace; no numeric coercion). `unit` = row's inherited unit. **Empty cells skipped.**

6. **Merged-cell handling.** Column span → value replicated across each spanned column. Row span → value replicated down.

7. **`raw_markdown` capture.** For every table including OTHER, capture the Docling-generated flat text representation. **Lookup rule (precise):**
   - **First preference:** scan `doc_json["texts"]` for any element whose `prov[0].$ref == "#/tables/{table_index}"` (Docling-graph convention for cross-references). Use that element's `text` field. If multiple matches, use the first.
   - **Second preference:** `doc_json["tables"][table_index].get("text", "")` if non-empty.
   - **Final fallback:** `doc_json["tables"][table_index].get("data", {}).get("table_markdown", "")`, or `""` if absent. Logged at DEBUG when this fires — operational signal that the table's flat-text mirror is missing.

**Error handling.** Per-table exception caught, logged at WARNING with `table_index` + `self_ref`, produces `NormalizedTable(shape=OTHER, cells=(), raw_markdown=...)` for that table. Other tables continue. No table failure breaks the pipeline.

**Idempotency.** `normalize_tables(doc_json)` reads but never writes `doc_json`. Callers cache the result per pass.

**Diagnostics** (routed to `diagnostics["service_table_normalization"]` — peer to existing `service_table_overlay`):
```python
{
    "tables_seen": 12,
    "tables_normalized": 9,
    "tables_skipped_other": 3,
    "tables_failed": 0,
    "shapes_by_type": {"column_major": 5, "row_major": 1, "hybrid": 3, "other": 3},
    "empty_cells_skipped": 47,
    "merged_cells_expanded": 18,
    "normalization_failures": [],
    "other_with_dimensions_warning": [...],   # tables ≥4×4 that fell to OTHER (operational signal)
}
```

## 9. Graph Renderer (`render_graph.py`)

```python
def render_for_graph(
    table: NormalizedTable,
    token_limit_whole: int,                 # caller passes from config (DOCLING_GRAPH_TABLE_WHOLE_LIMIT; default 1500)
    token_limit_column: int,                # caller passes from config (DOCLING_GRAPH_TABLE_COLUMN_LIMIT; default 1200)
) -> list[GraphTableChunk]: ...
```

(Function-signature defaults removed deliberately — same rationale as
`render_for_embedding`. `table_normalization/config.py` is the single source
of truth for thresholds.)

```python
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
```

**Decision tree:**

1. Shape == `OTHER` → one `TABLE_WHOLE` chunk with `text = raw_markdown`, empty cell_refs. No table disappears.
2. Whole-table rendering ≤ `token_limit_whole` → one structured `TABLE_WHOLE` chunk.
3. Whole-table > `token_limit_whole` → per-column emission:
   - For each `NormalizedColumn`: render as one chunk (`TABLE_ENTITY_COLUMN`).
   - If single-column rendering > `token_limit_column`: split by `TableSection`, each section chunk repeating the column's identity header (`TABLE_ENTITY_SECTION`).

**Chunk text format (uppercase section names, `GENERAL:` for unsectioned rows):**

```
TABLE: <caption or table_ref>
SOURCE: page <pages>

ENTITY:
- Industry Designation: <value>
- Military Designation: <value>
- NATO Designation: <value>
- Fan Song Variant: <value>

<SECTION 1 NAME>:
- <row_label>: <value> <unit>

<SECTION 2 NAME>:
- <row_label>: <value> <unit>
```

Multi-value cells (`"1135/1028"`) rendered as-is. Empty values skipped. Raw markdown of the source table is **not** included.

**Token measurement** via `tokens.py` (BAAI/bge-m3 tokenizer, lazy-loaded once per process).

### 9.1 Integration in `docker/docling-graph/app/main.py`

Replaces the current behavior at the integration point (after sanitization, before `run_extraction_pass`). The flag matrix has three orthogonal switches:

| `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED` | `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` | Behavior |
|---|---|---|
| `false` | `false` | **Today's production behavior, byte-identical.** Raw `texts[]` unchanged; only `extract_table_overlay` runs. |
| `false` | `true` | Experimental path — `synthesize_table_facts` from `_table_facts.py` runs on raw `texts[]`. (A/B for the reverted experiment.) |
| `true` | `false` | **New default** — normalize, render per-column, suppress raw flattened table mirrors. |
| `true` | `true` | **Disallowed.** Logged at ERROR; falls back to row-1 behavior. (The two paths emit conflicting overlays.) |

```python
normalized = normalize_tables(doc_json) if _norm_enabled() else []

if _norm_enabled() and _experimental_enabled():
    logger.error(
        "Both DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED and "
        "DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS are true; falling back to "
        "off+off (today's production behavior)."
    )
elif _norm_enabled():
    # New default path.
    for nt in normalized:
        for gtc in render_for_graph(nt):
            doc_json["texts"].append(_text_item_from_chunk(gtc))
    if _suppress_raw_enabled():
        _suppress_raw_table_texts(doc_json, normalized)
elif _experimental_enabled():
    synthesize_table_facts(doc_json, ...)   # _table_facts.py legacy path
else:
    pass   # today's production behavior: nothing changes in texts[]
```

### 9.2 Invariant: `_suppress_raw_table_texts` blanks in place; does NOT remove or reindex

**Critical:** `main.py:379` documents an existing pipeline invariant — *"Removing an element shifts all subsequent indices, but the [sanitizer] blanks content"*. Following the same pattern, suppression **blanks the text in place** rather than removing list entries. This preserves `self_ref` stability throughout `doc_json` (children references, parent backlinks, cell `prov` entries, OCR provenance — all of which may carry numeric `#/texts/N` pointers).

```python
def _suppress_raw_table_texts(doc_json: dict, normalized: list[NormalizedTable]) -> None:
    """Blank flat-text mirrors of normalized tables in-place.

    Invariants (asserted by unit test):
    - len(doc_json['texts']) is UNCHANGED. No element is removed; no index shifts.
    - doc_json['tables'] is NOT touched. The Phase 0/0.5 overlay machinery
      (extract_table_overlay → table_overlay.py merge-time application) reads
      from doc_json['tables'] directly and must remain functional.
    - Tables with shape == OTHER keep their flat text — the embedding-side
      OTHER fallback and the graph-side OTHER fallback both depend on the
      original text remaining visible.
    - For texts[] entries whose self_ref matches '#/tables/{i}' for some i
      with NormalizedTable.shape != OTHER: set entry['text'] = "" and
      entry['orig'] = "" (matches main.py:379 sanitizer pattern).
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    target_prefixes = tuple(f"#/tables/{i}" for i in non_other)
    for t in doc_json["texts"]:
        if t.get("self_ref", "").startswith(target_prefixes):
            t["text"] = ""
            t["orig"] = ""
```

## 10. Embedding Renderer (`render_embedding.py`)

```python
def render_for_embedding(
    table: NormalizedTable,
    token_limit: int,                       # caller passes settings.embedding_chunk_max_tokens (app/config.py:400; default 512)
    summary_limit: int,                     # caller passes settings.embedding_table_summary_max_tokens (new; default 300)
) -> list[EmbeddingTableChunk]: ...
```

(Function-signature defaults removed deliberately. The single source of truth
for these values is `app/services/table_normalization/config.py`, which reads
the env vars. Hardcoded signature defaults would silently drift if env values
were re-tuned.)

```python
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

**Emission rules:**

1. Shape == `OTHER` → one `TABLE_WHOLE` chunk with `text = raw_markdown`. Today's behavior.
2. **Always emit `TABLE_SUMMARY`** (one per table; capped at `summary_limit`):
   ```
   TABLE: <caption>
   SOURCE: page <pages>; ref <table_ref>
   VARIANTS: <display_names; truncated with "..." if too many>
   PROPERTIES: <spec-row labels; identity/section excluded; truncated similarly>
   ```
3. Whole-table rendering ≤ `token_limit` → also emit `TABLE_WHOLE` chunk.
4. Whole-table > `token_limit` → per-column chunks (`TABLE_ENTITY_COLUMN`); if a column > `token_limit`, split by section (`TABLE_ENTITY_SECTION`). `TABLE_WHOLE` **not** emitted.

**Shared rendering helper:** `_render_column_as_text(column, table, sections)` — produces the identity+sections+rows block, byte-identical between graph and embedding renderers for the same column input. Snapshot test asserts this.

### 10.1 Integration in `app/workers/pipeline.py` (primary path — HybridChunker)

**Where:** post-process `native_chunks` after `list(chunker.chunk(doc_obj_dl))` at `pipeline.py:5534` and before the chunk-iteration loop at `:5559`.

The integration replaces each native chunk that represents a normalized table with the chunks from `render_for_embedding(nt)`. Native chunks for tables we *can't* normalize (Shape.OTHER) or non-table content pass through unchanged.

```python
# pipeline.py — between line 5534 and line 5553

if use_native_chunking:
    if is_table_normalization_enabled_embedding():
        # Build normalized table map once per document.
        from app.services.table_normalization import normalize_tables, render_for_embedding
        from app.services.table_normalization.config import (
            embedding_chunk_max_tokens, embedding_table_summary_max_tokens,
        )
        normalized_by_table_idx = {
            nt.table_index: nt for nt in normalize_tables(doc_dict)
        }
        native_chunks = _substitute_table_chunks(
            native_chunks,
            normalized_by_table_idx,
            render_for_embedding,
            token_limit=embedding_chunk_max_tokens(),
            summary_limit=embedding_table_summary_max_tokens(),
        )
```

```python
# new helper in pipeline.py or a new app/services/table_normalization/_pipeline_hooks.py

def _substitute_table_chunks(
    native_chunks: list,
    normalized_by_table_idx: dict[int, NormalizedTable],
    render_fn,
    token_limit: int,
    summary_limit: int,
) -> list:
    """Replace native HybridChunker table chunks with normalized chunks.

    A native chunk is treated as a 'table chunk' when ANY doc_item it
    references has a self_ref matching '#/tables/{i}' for some i in
    normalized_by_table_idx whose NormalizedTable has shape != OTHER.

    For such chunks, render_fn is called on the NormalizedTable and the
    resulting EmbeddingTableChunks replace the native chunk in place.
    Multiple native chunks pointing to the same table index produce only
    ONE substitution (the first occurrence); subsequent native chunks for
    the same table are dropped — the normalized chunks already cover all
    its content.
    """
    seen_table_idx: set[int] = set()
    out: list = []
    for nc in native_chunks:
        table_idx = _table_idx_for_native_chunk(nc, normalized_by_table_idx)
        if table_idx is None:
            out.append(nc)               # not a normalizable-table chunk; preserve
            continue
        if table_idx in seen_table_idx:
            continue                     # already substituted for this table; drop
        seen_table_idx.add(table_idx)
        nt = normalized_by_table_idx[table_idx]
        for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
            out.append(_NormalizedTableChunkAdapter(etc))  # ducktypes the HybridChunker chunk interface (see below)
    return out


def _table_idx_for_native_chunk(nc, normalized_by_table_idx) -> int | None:
    """Return the matching normalized-table index if this native chunk
    references a normalized table whose shape != OTHER; else None.

    Iterates nc.meta.doc_items; for each, parses self_ref of form
    '#/tables/N' to extract N; returns the first match in
    normalized_by_table_idx (or None)."""
    for item in (getattr(getattr(nc, "meta", None), "doc_items", None) or []):
        ref = getattr(item, "self_ref", None) or ""
        if not ref.startswith("#/tables/"):
            continue
        try:
            idx = int(ref.split("/")[-1])
        except (ValueError, IndexError):
            continue
        if idx in normalized_by_table_idx and normalized_by_table_idx[idx].shape != Shape.OTHER:
            return idx
    return None
```

**The adapter (`_NormalizedTableChunkAdapter`)** wraps an `EmbeddingTableChunk` to satisfy the duck-typed interface the rest of `pipeline.py:5559-5623` reads off native chunks: `.text`, `.meta.doc_items[].self_ref`, `.meta.doc_items[].prov[].page_no`, `.meta.headings`. The adapter populates these from the underlying `EmbeddingTableChunk`:
- `.text` → `etc.text`.
- `.meta.doc_items` → synthetic items whose `self_ref` is each entry in `etc.cell_refs` (so existing `_build_native_chunk_meta` at `pipeline.py:5411-5420` produces correct `self_refs`).
- `.meta.doc_items[].prov[].page_no` → page numbers from `etc.page_numbers`.
- `.meta.headings` → `[etc.table_caption]` if caption present, else `[]`.

The existing `_build_native_chunk_meta` then produces the correct `self_refs`, `page_numbers`, `evidence_ids`, and `section_path` *without modification*. The new metadata fields (`chunk_kind`, `table_ref`, `entity_display_name`, `section`, `column_index`, `row_labels`) are passed separately: `_build_native_chunk_meta` gains an optional parameter `extra_metadata: dict | None = None` that, when present, populates the returned dict's `chunk_metadata` key, which then flows into the new Postgres column.

```python
# pipeline.py:5435 modification — add chunk_metadata to the return dict
return {
    "chunk_id": chunk_id,
    "chunk_index": chunk_idx,
    ...
    "chunk_metadata": extra_metadata,   # new
}
```

And `pipeline.py:5594` modification — write the column:
```python
chunk_values = {
    ...
    "bounding_box": None,
    "chunk_metadata": meta.get("chunk_metadata"),   # new
}
```

ArcadeDB write at `pipeline.py:5603-5618` adds the `chunk_kind` property to `properties`:
```python
properties={
    ...
    "section_path": meta.get("section_path"),
    "headings": meta.get("headings", []),
    "chunk_kind": (meta.get("chunk_metadata") or {}).get("chunk_kind"),   # new
},
```

### 10.2 Integration in `app/services/chunking.py` (legacy fallback path)

When `use_native_chunking is False` at `pipeline.py:5636-5664` (HybridChunker failed or no enrichment), the legacy `structure_aware_chunk` runs. We integrate there as well — the legacy path is rarer but real.

```python
def is_table_normalization_enabled_embedding() -> bool:
    # Default "false" matches §12 (master switches ship disabled).
    return os.environ.get("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "false").lower() == "true"


# In structure_aware_chunk(...) — replace lines 113-126:
elif etype == "table":
    current_heading = buffer_heading
    _flush_buffer()
    nt = normalized_table_for(elem, normalized_tables) if is_table_normalization_enabled_embedding() else None
    if nt is None or nt.shape == Shape.OTHER:
        # Master kill-switch OR un-normalizable shape → today's behavior.
        chunks.append(StructuredChunk(text=content, chunk_index=chunk_index,
                                      modality="table", page_number=page,
                                      section_path=section, element_uids=[uid],
                                      heading_text=current_heading))
        chunk_index += 1
    else:
        for etc in render_for_embedding(nt, ...):
            chunks.append(StructuredChunk(
                text=etc.text,
                chunk_index=chunk_index,
                modality="table",
                page_number=etc.page_numbers[0] if etc.page_numbers else None,
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

`StructuredChunk` gains optional `metadata: dict | None = None` (kw-only with default; existing call sites unchanged).

**`normalized_table_for(elem, normalized_tables)`** — lookup function from a `DocumentElement` dict to a `NormalizedTable`. The bridge: `DocumentElement.element_uid` doesn't carry the table index directly, but `DocumentElement.metadata` (a JSONB field already on the model) stores the docling self_ref for elements derived from docling. The function:

```python
def normalized_table_for(elem: dict, normalized_tables: list[NormalizedTable]) -> NormalizedTable | None:
    # Try direct self_ref in element metadata (preferred).
    ref = (elem.get("metadata") or {}).get("self_ref", "")
    if ref.startswith("#/tables/"):
        try:
            idx = int(ref.split("/")[-1])
        except (ValueError, IndexError):
            return None
        return next((nt for nt in normalized_tables if nt.table_index == idx), None)
    # Legacy fallback: match by content_text fuzzy heuristic — log warning.
    logger.debug("normalized_table_for: element has no self_ref; using None.")
    return None
```

If `DocumentElement.metadata` does NOT carry `self_ref` today, the **implementation spike at §20** must verify and either (a) add `self_ref` to the docling-import path so the metadata field is populated, or (b) introduce an alternative bridge (e.g., match by `content_text` substring). The legacy path's success depends on this; the spike resolves before any other code lands.

### 10.3 Master kill-switch correctness

With `EMBEDDING_TABLE_NORMALIZATION_ENABLED=false`:
- **Primary path (HybridChunker):** the `if is_table_normalization_enabled_embedding():` block at `pipeline.py:5534` is skipped. `native_chunks` flow unchanged into the existing loop. **Byte-identical to today.**
- **Legacy path:** `normalized_table_for` returns `None` because the flag is off (function short-circuits when flag is false), falling through to today's `StructuredChunk(text=content, ...)` call. **Byte-identical to today** (the `metadata=None` default on `StructuredChunk` is not written to DB; see §14 test note on what "byte-identical" precisely means).

**Test note on byte-identity:** the test compares the serialized list of `(chunk_text, modality, page_number, section_path, element_uids)` tuples against the §19 baseline fixture. `StructuredChunk` repr is not used — the new `metadata` field's `__repr__` is irrelevant. The new `chunk_metadata` JSONB column on `TextChunk` stays `NULL` when the flag is off. Documented in `test_master_kill_switch_byte_equality.py`.

## 11. Provenance & Storage

### 11.1 Alembic migration

```python
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
    # WARNING: drops the chunk_metadata column and all provenance data it contains.
    # Run ./manage.sh --blow-away before downgrading to avoid silent data loss in
    # retrieval responses that rely on table_chunk metadata.
    op.drop_index("ix_text_chunks_chunk_kind", table_name="text_chunks", schema="retrieval")
    op.drop_column("text_chunks", "chunk_metadata", schema="retrieval")
```

Additive nullable column. Partial expression index on `chunk_kind` for future retrieval-side filtering / scoring. Index is cheap because most chunks have `chunk_metadata IS NULL` (prose chunks).

### 11.2 `chunk_metadata` payload schema

For normalized table chunks:
```json
{
  "chunk_kind": "table_entity_column",
  "table_ref": "#/tables/3",
  "entity_display_name": "S-75M2 / SA-2D",
  "section": null,
  "column_index": 7,
  "cell_refs": ["#/tables/3/data/table_cells/42", ...],
  "row_labels": ["Max Range", "Min Range", "Max Altitude"],
  "table_caption": "S-75 Technical Data ...",
  "page_numbers": [6, 7]
}
```

For non-table chunks: `chunk_metadata` stays `NULL`.

### 11.3 Model update (only DB-schema change)

```python
class TextChunk(Base, TimestampMixin):
    # ... existing fields ...
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
```

(In-memory `StructuredChunk` also gains a `metadata` field — see §10.1 — but that is not a DB schema change.)

### 11.4 ArcadeDB mirroring

`TextChunk` vertices in ArcadeDB gain one new property: `chunk_kind: str | None`. The existing schema-bootstrap pattern in `arcadedb_schema.py:31` is a property tuple list per vertex type. The change is **one line**:

```python
# arcadedb_schema.py:31
"TextChunk": [
    ("chunk_id", "STRING"),
    ("document_id", "STRING"),
    ("page_number", "INTEGER"),
    ("modality", "STRING"),
    ("classification", "STRING"),
    ("text_embedding", "ARRAY_OF_FLOATS"),
    ("chunk_kind", "STRING"),            # NEW
],
```

The bootstrap loop at `arcadedb_schema.py:199` runs `CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {prop_type}` — idempotent, safe on existing populated vertex types.

**Index addition:** the existing index-creation pattern in `arcadedb_schema.py` (around line 280 for `text_embedding`) is per-type. For `chunk_kind`, add to the appropriate index loop:

```python
# After existing TextChunk index declarations
self._exec(f"CREATE INDEX IF NOT EXISTS ON TextChunk (chunk_kind) NOTUNIQUE")
```

Both `CREATE PROPERTY ... IF NOT EXISTS` and `CREATE INDEX ... IF NOT EXISTS` are idempotent. Either fresh deploy (`./manage.sh --blow-away`) or in-place restart picks up the new schema. Production write paths (`arcadedb_client.py` / `arcadedb_graph.py`) write `chunk_kind` from `meta["chunk_kind"]` per §10.1 (`properties` dict on the `TextChunkRecord`).

### 11.5 Retrieval surfacing (`/v1/retrieval/query`)

Response envelope gains an optional `table_chunk` block per result when `chunk_metadata IS NOT NULL`:

```json
{
  "text": "...",
  "document_id": "...",
  "page_numbers": [6, 7],
  "self_refs": ["#/texts/142"],
  "evidence_ids": [...],
  "table_chunk": {
    "kind": "table_entity_column",
    "table_ref": "#/tables/3",
    "table_caption": "...",
    "entity_display_name": "S-75M2 / SA-2D",
    "section": null,
    "cell_refs": [...],
    "row_labels": [...]
  }
}
```

Additive; backwards-compatible.

### 11.6 Graph-side provenance wiring

**The walker:** `docker/docling-graph/app/provenance.py` (596 lines). It reads two extraction-time maps built at `main.py:765-781`:
- `chunk_to_self_refs: dict[int, list[str]]` — chunk_index → list of `self_ref` strings.
- `chunk_to_evidence_units: dict[int, list[dict]]` — chunk_index → list of evidence-unit dicts.

The walker's `_resolve_element_uid` (lines 175-223) resolves a node's `element_uid` via a 5-step priority:
1. Direct `element_uid` on the node.
2. `provenance.element_uid`.
3. `provenance.self_refs[0]` — **authoritative when set by the library**.
4. `provenance.evidence_ids[0]` if it starts with `'#/'`.
5. `provenance.chunk_indexes[0]` → first `self_ref` via `chunk_to_self_refs`.

**The flow we need:** when an entity is extracted from one of our `GraphTableChunk`-derived TextItems, its `provenance.self_refs` (and `evidence_ids`) must include the underlying `cell_refs`.

**Mechanism:** `_text_item_from_chunk(gtc)` (the helper in the new module that converts a `GraphTableChunk` to a docling `TextItem`) constructs the TextItem so:
- The TextItem's own `self_ref` is `#/texts/{N}` (synthetic, assigned at append time by docling's normal mechanism).
- The TextItem's `prov: list[ProvenanceItem]` is populated with entries pointing to each cell in `gtc.cell_refs` — each `prov` entry's `$ref` is set to the cell's self_ref.
- When the graph-extraction library records this chunk's creation, it populates `chunk_to_self_refs[chunk_index]` with both the synthetic TextItem's `self_ref` AND the cell refs from `prov[].$ref`. (This is how the library handles tables today — the prior `_table_facts.py` work used the same mechanism for its synthesized items; see `_table_facts.py:823` for the `prov` construction pattern.)

**What this assumes:** the graph-extraction library indexes `prov[].$ref` into `chunk_to_self_refs[chunk_index]` when building the chunk-creation trace. This is the load-bearing assumption.

**§20 spike resolves this assumption before any other code lands.** The spike: construct a single TextItem with `self_ref` + `prov` referencing a known cell, run it through a single extraction pass on a fixture, inspect `chunk_to_self_refs` and the resulting `ExtractionProvenance.evidence_ids`. If the cell ref appears, ship the spec as written. If it doesn't, fix the gap — likely options: (a) extend `_chunk_maps_from_trace` (`main.py:953`) to flatten `prov[].$ref` into the per-chunk ref list, or (b) post-process `chunk_to_self_refs` to merge in our cell refs before the walker runs. The spike's outcome determines which.

**`test_graph_provenance_cell_refs.py`** is the merge-gate test: extract an entity from a SA-2 column chunk; assert its `ExtractionFieldProvenance.evidence_ids` contains a `#/tables/N/data/table_cells/M` cell ref.

## 12. Configuration

| Variable | New / Existing | Default | Side | Effect |
|---|---|---|---|---|
| `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED` | **new** | `false` | Graph | Master switch. **Default false to ship without changing today's production behavior.** |
| `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` | **new** | `false` | Graph | Experimental — run reverted `_table_facts.py`. Mutually exclusive with the above. |
| `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN` | **new** | `true` | Graph | When normalization is on, strip flat-text mirrors of normalized tables. |
| `DOCLING_GRAPH_TABLE_WHOLE_LIMIT` | **new** | `1500` | Graph | Whole-vs-column threshold (tokens). |
| `DOCLING_GRAPH_TABLE_COLUMN_LIMIT` | **new** | `1200` | Graph | Split-column-by-section threshold (tokens). |
| `EMBEDDING_TABLE_NORMALIZATION_ENABLED` | **new** | `false` | Embedding | Master switch for embedding side. **Default false.** |
| `EMBEDDING_CHUNK_MAX_TOKENS` | **existing** (`app/config.py:400`) | `512` | Embedding | Plumbed into `table_normalization/config.py`; matches `bge-large-en-v1.5` hard limit. |
| `EMBEDDING_TABLE_SUMMARY_MAX_TOKENS` | **new** | `300` | Embedding | Cap on summary chunk. |

**Net new variables: 7.** One reused. All seven new vars land in `.env` and `.env.example` with default + one-line comment, per `feedback_env_vars_must_appear_in_dotenv_files`.

**Default-off note (changed from rev. 1):** the master switches now default to `false`, not `true`. This ships the code without changing production behavior. Enabling is a separate operational step *after* the baseline-capture procedure in §19 completes. Rationale: per `feedback_post_code_workflow`, code lands first, behavior changes after verification.

## 13. Rollout

**One PR, but two operational phases:**

### Phase 1 — Code lands, behavior unchanged

1. **Step 0a (BEFORE any code changes)** — capture baseline. See §19.
2. **Step 0b** — run the implementation spike (§20). Resolves provenance-flow + legacy-element-bridge assumptions. May add a small fix (~10–25 LOC) before feature work.
3. Apply alembic migration (additive nullable column + partial index).
4. Feature code merges to `main` with all master switches at `false` defaults. Production behavior is byte-identical to pre-merge.
5. Run integration test `test_master_kill_switch_byte_equality.py` against the SA-2 corpus to confirm both HybridChunker and legacy paths emit unchanged outputs vs. §19 baseline. **This is a merge gate.**

### Phase 2 — Behavior changes, gated on regression check

5. Flip `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true` + `EMBEDDING_TABLE_NORMALIZATION_ENABLED=true`. Run `./manage.sh --blow-away` and re-ingest.
6. Score extraction against the §19 `sa2_<docid>_extraction_today.json` baseline fixtures (the same fixtures captured in Step 0 — there is only one baseline, not two).
7. **Flip gate (§15.2):** new path must satisfy *all* the conditions in §15.2 before the flip is considered shipped. If any condition fails: flip flags back to `false` (cost: one `--blow-away` + re-ingest), confirm rollback by re-running `test_master_kill_switch_byte_equality.py` against the §19 baseline, file a follow-up issue with comparison data.

### A/B experimentation (post-flip, optional)

- Set `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` (and turn the new-path master switch off) to run the reverted `_table_facts.py` codepath. Useful for comparison data; not a rollback target.

### Container rebuild posture
- `app/services/table_normalization/` is under bind-mounted `./app` — no rebuild for `api`/`worker`/`worker-graph`.
- `docker compose build docling-graph` required to pick up the new module (COPY into image).

## 14. Test Posture

| Layer | Coverage | Location |
|---|---|---|
| Detection unit | SA-2 hybrid; column-major; row-major; undersized 3×3 → OTHER; 4×4 boundary; malformed flags; section headers; OTHER-with-dimensions warning emitted | `tests/unit/test_table_normalization_detect.py` |
| Normalization unit | Empty cells skipped; merged cells expanded; units extracted; multi-value cells preserved; idempotency; raw_markdown source resolution; row-major transpose | `tests/unit/test_table_normalization_normalize.py` |
| Graph renderer unit + snapshot | SA-2 column chunks (≥1 chunk per entity column); small whole; long-column section-split; OTHER fallback; format identical to embedding helper | `tests/unit/test_table_normalization_render_graph.py` + JSON fixture |
| Embedding renderer unit + snapshot | Always-summary; size-aware whole/column; format identical to graph; OTHER fallback | `tests/unit/test_table_normalization_render_embedding.py` + JSON fixture |
| Shared helper byte-equality | `_render_column_as_text` produces byte-identical output called from either renderer for the same column | `tests/unit/test_render_column_byte_equality.py` |
| Token sizing | bge-m3 tokenizer agrees with chunk-size budget within ±5% | `tests/unit/test_table_normalization_tokens.py` |
| Suppression invariant | `_suppress_raw_table_texts` mutates only `doc_json["texts"]`; `doc_json["tables"]` byte-identical pre/post; OTHER tables' texts preserved | `tests/unit/test_suppress_raw_table_texts_invariant.py` |
| Master kill-switch byte equality | With both `*_NORMALIZATION_ENABLED=false`, on SA-2: (a) `doc_json["texts"]` after sanitization is byte-identical to §19 fixture; (b) HybridChunker path produces the same `(chunk_text, modality, self_refs, page_numbers)` tuples as today on the same input; (c) legacy `structure_aware_chunk` path produces the same `StructuredChunk` field tuples as today. The new `metadata` field on `StructuredChunk` is `None` in this state and is excluded from the comparison. | `tests/integration/test_master_kill_switch_byte_equality.py` |
| HybridChunker substitution unit | `_substitute_table_chunks` replaces native table chunks with `EmbeddingTableChunk` outputs when shape != OTHER; preserves non-table chunks; drops duplicate native chunks for same table; preserves OTHER tables unchanged | `tests/unit/test_hybrid_chunker_substitution.py` |
| Adapter ducktyping | `_NormalizedTableChunkAdapter` produces `.text`, `.meta.doc_items[].self_ref`, `.meta.doc_items[].prov[].page_no`, `.meta.headings` such that existing `_build_native_chunk_meta` returns the correct `self_refs`/`page_numbers`/`section_path` | `tests/unit/test_normalized_table_chunk_adapter.py` |
| Storage round-trip | Write/read `chunk_metadata`; partial index returns matching rows | `tests/integration/test_chunk_metadata_persistence.py` |
| Retrieval surfacing | `/v1/retrieval/query` returns `table_chunk` block when `chunk_metadata` non-null; absent otherwise | `tests/integration/test_retrieval_table_chunk_surfacing.py` |
| Graph provenance | `ExtractionFieldProvenance.evidence_ids` includes underlying cell refs for SA-2 propulsion extraction | `tests/integration/test_graph_provenance_cell_refs.py` |
| End-to-end SA-2 | Ingest → extract → score → cell_refs traceable | `tests/integration/test_sa2_table_pipeline_e2e.py` |
| Experimental-path drift guard | `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` produces non-empty `synthesize_table_facts()` output on SA-2 fixture. **Assertion: count of `TextItem`s emitted > 0 AND `TextItem` count is within ±10% of a recorded fixture value.** Count-based, not text-content snapshot — tolerates whitespace/cosmetic drift but catches catastrophic rot (the path stops emitting anything, or emits 10× too much). **This guards against `_table_facts.py` rotting on disk; it is NOT a baseline-matching test.** | `tests/integration/test_experimental_table_facts_drift.py` |
| Disallowed-combination fallback | Both `*_ENABLED` master flags AND `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` simultaneously → falls back to today's raw-blob behavior (per §9.1 row 4). Asserted: `texts[]` byte-identical to §19 baseline; ERROR log emitted. | `tests/integration/test_disallowed_combination_fallback.py` |

**Additions to `VERIFICATION_CHECKLIST.md`:**
- `□ Step 0a baseline fixtures committed under tests/fixtures/sa2/ (main_sha recorded in baseline.meta.json)`
- `□ Step 0b spike completed: provenance-flow and legacy-element-bridge tests pass (with fixes from §20 applied if needed)`
- `□ Phase 1 merge: master kill-switch byte-equality test passes against §19 baseline (both HybridChunker and legacy paths)`
- `□ Phase 2 flip: SA-2 missile_propulsion ✓ exact ≥ today-baseline ✓ exact (§15.2)`
- `□ Phase 2 flip: missile_propulsion wrong count ≤ today-baseline wrong + 1`
- `□ Phase 2 flip: no other pass regresses by >1 ✓ exact (per-pass)`
- `□ Phase 2 flip: corpus-wide ✓ exact sum ≥ today-baseline sum − 2`
- `□ Phase 2 flip: variance-mode (strict / median) decision recorded in baseline.meta.json`
- `□ .env + .env.example contain all 7 new variables`

## 15. Success Criteria (acceptance gates)

### 15.1 Phase 1 merge gate (code-merge time)

- All unit tests pass.
- `test_master_kill_switch_byte_equality.py` proves both sides emit byte-identical chunks vs. the §19 baseline with master switches off.
- `test_suppress_raw_table_texts_invariant.py` proves the `tables[]`-untouched invariant.
- `test_render_column_byte_equality.py` proves both renderers share format.

### 15.2 Phase 2 flip gate (production-flip time)

All conditions must hold. Each is checked against the §19 baseline fixture.

**Extraction-quality gates (all required):**
- SA-2 `missile_propulsion` ✓ exact ≥ today-baseline ✓ exact (strict no-regression on the pass the new approach is most likely to affect).
- SA-2 `missile_propulsion` `wrong` count ≤ today-baseline `wrong` count + 1 (guards the case where today-baseline is at ✓0 floor and ✓ exact comparison is trivially satisfied).
- **Per-pass tolerance (tightened from rev. 2):** at most ONE non-propulsion pass may regress by exactly 1 ✓ exact. All other non-propulsion passes must satisfy new ✓ exact ≥ today-baseline ✓ exact. (Rev. 2 allowed every pass to regress by 1 individually, which compounded badly — see review I-2.)
- **Corpus-wide guard:** sum of ✓ exact across all passes ≥ sum of today-baseline ✓ exact − 1 (combined with the per-pass rule above, allows at most one ✓-loss across the corpus).

**Retrieval gate (relaxed from rev. 2 "top-1" to top-3, less brittle to embedding-score variance):**
- `/v1/retrieval/query` for *"S-75M2 max range"* returns *within top 3 results* at least one chunk with `chunk_kind == "table_entity_column"`, `entity_display_name` containing "S-75M2", and `cell_refs` pointing to cells in the SA-2 variants table.

**Provenance gate:**
- Extracted `Missile.max_range_m` on `S-75M2` has `ExtractionFieldProvenance.evidence_ids` containing both the graph-chunk self_ref AND the underlying cell ref.

### 15.3 Functional (informational, not merge-gating)

- SA-2 sample table → `Shape.HYBRID`, 4 identity rows, 2+ sections, all data columns extracted to `NormalizedColumn`.
- Graph renderer for SA-2 emits ≥1 chunk per entity column (count depends on whether columns exceed `DOCLING_GRAPH_TABLE_COLUMN_LIMIT`; assertion is "≥1 per column," not a snapshot count).
- Embedding renderer for SA-2 emits 1 `TABLE_SUMMARY` + N `TABLE_ENTITY_COLUMN` chunks. `TABLE_WHOLE` not emitted (table > 512 tokens — verified by token sizing test).

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **No verified baseline for the column-arithmetic failure mode.** The prior empirical claim (✓0 → ✓≥6 at T=1.0) was unverified at the time `_table_facts.py` was reverted. We don't know how much worse, if at all, today's raw-flattened path performs on SA-2 propulsion compared to either alternative. | Capture today-baseline extraction outputs *before* any flip (§13 Phase 2 step 5). The merge gate measures *against today's actual production output*, which is the only baseline that exists. |
| **New path may push column-arithmetic load back onto the LLM.** Per-column chunks with raw row labels are what the prior work documented as "did not fix" the off-by-one shift. New path may regress SA-2 propulsion vs today. | Phase 2 flip gate (§15.2) blocks production-flip if SA-2 `missile_propulsion` ✓ exact regresses. The flag flip is reversible (one `--blow-away` + re-ingest). |
| **Token-budget defaults may be wrong for non-SA-2 corpora.** SA-2 was the basis for the analysis; other docs may have different shapes/sizes. Defaults `1500/1200/512/300` are educated guesses, not measured. | All budgets are env-tunable. Diagnostics capture `tables_fit_whole`/`tables_split_by_column`/`by_kind`. After Phase 2 flip, measure across corpus and tune defaults if needed before merging. |
| **Shape detection misclassifies a real table as OTHER.** Heuristic floor (4×4) and reliance on Docling's `row_header`/`column_header` flags assume clean parsing. | OTHER fallback is *additive-not-destructive*: raw markdown preserved as one chunk. The `other_with_dimensions_warning` diagnostic surfaces 4×4+ tables that fell to OTHER for operational visibility. |
| **`_suppress_raw_table_texts` could break the Phase 0/0.5 overlay.** The merge-time overlay machinery reads `doc_json["tables"]` directly; the suppression only touches `doc_json["texts"]`. Spec asserts the invariant (§9.2). | `test_suppress_raw_table_texts_invariant.py` is a merge gate (§15.1). |
| **`_table_facts.py` + `_alias_map.py` rot on disk.** Code that doesn't run in production rots; the experimental path could become fictional. | `test_experimental_table_facts_drift.py` runs the experimental path on every CI build (§14). Failure surfaces drift. |
| **ArcadeDB `chunk_kind` property requires --blow-away.** Deployments skipping blow-away fail at vertex write. | Documented in §11.4 + verification checklist + .env.example. User's standard flow (`./manage.sh --blow-away`) covers this. |
| **`bge-m3` tokenizer load adds startup cost.** First call loads HF tokenizer (~tens of MB). | Lazy-load + module-level cache. Same model the embedding pipeline already pulls. Not measurable in steady-state. |

## 17. Known Limitations

- Row-major tables supported but not primary focus.
- `display_name` heuristic hand-coded (Industry → Military → NATO → Missile Type → `col-N`).
- Spec-row, section, identity keyword lists hand-coded (closed in §7).
- `cell_refs` JSON-Pointer format assumes Docling's `data.table_cells` array structure is stable. Mitigated by capturing `table_index` + `(row_idx, col_idx)` redundantly inside the pointer string.

## 18. Open Questions

None at design time. All decisions locked through brainstorming clarifying questions (2026-05-11 conversation) and refined through review (rev. 2).

## 19. Baseline Capture Procedure (Step 0)

**Must complete BEFORE any code changes land on `feat/table-aware-chunking`.**

### Inputs
- Current `main` HEAD (record SHA in `tests/fixtures/sa2_today_baseline.meta.json`).
- SA-2 corpus documents (whichever the user is using for `think_true_*.csv` runs).

### Procedure
1. On current `main` (no code changes from this branch yet), ingest the SA-2 corpus using the user's existing ingest command — the same one used for the `think_true_*.csv` runs.

   The capture procedure does NOT prescribe ingest mechanics — the user has an existing flow. The implementation plan must record the exact command used so this step is reproducible.

2. **Run the corpus N=3 times** (the LLM variance check determines whether the §15.2 gate compares against best-of-3, median-of-3, or strict single-run; see decision rule below). Each run must:
   - Reset to the same input state (e.g., `./manage.sh --blow-away` if needed; ingest cache cleared between runs to eliminate caching artifacts).
   - Use temperature=0 (or whichever temperature the user's `think_true_*.csv` runs use; record in `baseline.meta.json`).

   **Capture mechanism for `doc_json["texts"]`:** add a temporary instrumentation hook in `docker/docling-graph/app/main.py` immediately after sanitization (around line 564, before the reverted-block comment) that dumps `body.docling_document_json["texts"]` to disk:
   ```python
   if os.environ.get("CAPTURE_BASELINE_TEXTS", ""):
       _baseline_dir = os.environ["CAPTURE_BASELINE_TEXTS"]
       _doc_id = body.document_id  # or whatever the doc identifier is
       with open(f"{_baseline_dir}/{_doc_id}_texts_today.json", "w") as f:
           json.dump(body.docling_document_json["texts"], f, indent=2)
   ```
   The hook is removed (or stays off via env-flag default) before the rev. 4 code lands. Captured per run is fine; the captured `texts[]` should be deterministic per (doc, sanitizer-version) so all 3 runs produce the same file.

   **Capture mechanism for extraction counts:** the user already has a scoring script that produces `{exact, wrong, null}` counts for `think_true_*.csv`. Whatever that script is, run it for each pass × doc × N=3 and aggregate into the JSON shape below.
3. For each SA-2 document, capture and commit:
   - **`tests/fixtures/sa2/<docid>_texts_today.json`** — full `doc_json["texts"]` after sanitization but before `run_extraction_pass`. Deterministic (pure function of input + sanitizer); single run sufficient. This is the chunk-level baseline for `test_master_kill_switch_byte_equality.py` (§15.1).
   - **`tests/fixtures/sa2/<docid>_extraction_counts_today.json`** — per-pass `{exact, wrong, null}` counts only, across all 3 runs:
     ```json
     {
       "missile_propulsion": {"runs": [{"exact": 3, "wrong": 12, "null": 5}, ...]},
       "kinematics":         {"runs": [...]},
       "speed_timing":       {"runs": [...]},
       ...
     }
     ```
     **Raw extraction JSON is NOT committed** — too large for git. The counts file is the only artifact the §15.2 gate reads.
   - Optionally: `tests/fixtures/sa2/<docid>_extraction_full_today.json` written but `.gitignore`d for local debugging only.
4. Record in `tests/fixtures/sa2/baseline.meta.json`:
   ```json
   {
     "captured_at": "2026-05-11T...",
     "main_sha": "<git rev-parse HEAD>",
     "docling_graph_image_id": "<docker compose images docling-graph --format json | jq -r '.[].ID'>",
     "corpus_files": ["...", "..."],
     "temperature": 0,
     "runs_per_doc": 3,
     "comparison_mode": "median"   // or "best" or "strict" — see decision rule
   }
   ```
5. Commit fixtures on `feat/table-aware-chunking` with message `test(baseline): capture today's SA-2 production behavior pre-rewrite (sha <SHA>)`.

### Variance / comparison-mode decision rule

After Step 2 (3 runs):
- Compute per-pass per-doc `max(exact) − min(exact)` across the 3 runs.
- **If `max − min ≤ 1` for every (pass, doc) pair**: comparison mode = `strict` (use run-0 counts; §15.2 gates compare strict equality / tolerance against run-0).
- **If `max − min ≥ 2` anywhere**: comparison mode = `median` (use the median of 3 runs as the baseline; §15.2 gates apply to median values).
- Record the decision in `baseline.meta.json`. The Phase 2 flip (§13) likewise runs N=3 and compares against the recorded mode.

This eliminates the rev. 2 ambiguity where "strict vs median" was a fallback without a decision rule.

### Why this matters

Without this, the merge gate at §15.1 and the flip gate at §15.2 are aspirational. With it, both gates have concrete, version-controlled targets to assert against. The fixture also doubles as the regression target for any future change to the table-chunking layer.

### What's captured vs. what's not

- **Captured & committed:** `doc_json["texts"]` per doc (deterministic), per-pass ✓ exact / wrong / null counts across 3 runs, baseline metadata.
- **Not committed:** full extraction JSON per pass per run (too large; reproducible from the recorded `main_sha`).
- **Captured & gitignored:** optional debugging dumps for local inspection.

---

## 20. Implementation Spike (Task 0 — before any other implementation work)

Two assumptions in this spec are not verified against the codebase. They must be resolved by a small spike BEFORE any feature work:

### 20.1 Provenance flow spike

**Question:** does `prov[].$ref` on a synthesized docling `TextItem` flow into `chunk_to_self_refs[chunk_index]` during graph extraction?

**Procedure:**
1. Construct a single docling `TextItem` with `self_ref="#/texts/test"`, `text="..."`, `prov=[ProvenanceItem(ref="#/tables/0/data/table_cells/42", ...)]`.
2. Inject it into a minimal `doc_json` and run one extraction pass against a trivial fixture (any pass; just need the trace).
3. Read `context._chunk_to_self_refs` at the end of the pass.

**Expected:** `chunk_to_self_refs[chunk_index]` contains `["#/texts/test", "#/tables/0/data/table_cells/42"]`.

**If it doesn't:** the spec's §11.6 walker-wiring needs one of two fixes:
- **(a)** Extend `_chunk_maps_from_trace` (`main.py:953`) to flatten `prov[].$ref` into the per-chunk ref list. ~10 LOC; tested.
- **(b)** Post-process `chunk_to_self_refs` after extraction-time-map construction, before the walker runs, to merge in cell refs for our synthesized TextItems. ~15 LOC; tested. (Slightly more invasive but doesn't touch the library-facing path.)

The spike's outcome determines which. Either fix is small.

### 20.2 Legacy-path element bridge spike

**Question:** does `DocumentElement.metadata` carry the docling `self_ref` for elements derived from docling tables today?

**Procedure:** ingest one document via the legacy path; query `SELECT id, element_type, metadata FROM ingest.document_elements WHERE element_type = 'table' LIMIT 5`. Inspect the `metadata` JSONB.

**Expected:** each table element has a `metadata.self_ref` matching `#/tables/N`.

**If it doesn't:** either (a) add `self_ref` to the docling-import path in `app/services/docling_anchors.py` so the metadata field is populated (small change; one place), or (b) match by `content_text` substring in `normalized_table_for` (brittle; only viable if (a) is infeasible). The spike determines which.

### Spike output

A single PR or commit on `feat/table-aware-chunking` that:
1. Adds a `tests/spike/test_provenance_flow.py` and `tests/spike/test_legacy_element_bridge.py`.
2. Both tests assert the expected behavior.
3. If either fails: includes the minimal fix from §20.1 (a)/(b) or §20.2 (a)/(b) above, with the test now passing.

This spike is the first task in the implementation plan. Subsequent feature work depends on its outcome. Estimated time: 1–2 hours for both spikes combined.
