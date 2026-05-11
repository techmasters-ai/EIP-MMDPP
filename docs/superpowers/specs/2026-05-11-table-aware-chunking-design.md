# Table-Aware Chunking — Design

**Status:** Draft 2026-05-11 (rev. 2 after review)
**Branch:** `feat/table-aware-chunking`
**Related:**
- `2026-05-05-section-aware-table-fact-synthesis-design.md` — prior spec for `_table_facts.py`, which was **built, then reverted from production** on 2026-05-06 (see §1 below). Modules remain on disk; not currently called by `run_extraction_pass`.
- `2026-05-06-table-identity-rewrite-and-field-overlay-design.md` — `table_overlay.py` Phase 0/0.5 wiring. **Untouched** by this spec.

---

## 1. Problem

### 1.1 What runs in production today

Reading `docker/docling-graph/app/main.py:566-574`:

> *"Section-aware table-fact synthesis (table_facts.py + alias_map.py) was built and validated in the 2026-05-06 plan, **then reverted here** after cross-pass measurement showed the cost (+10-30% wall on docs with variants tables, +output truncation pressure) outweighed the benefit (+2 ✓ exact on airframe for 1 of 21 corpus docs; no improvement on kinematics/speed_timing; **propulsion fix landed but unverified**). Modules remain on disk in app/_table_facts.py + app/_alias_map.py with full tests; re-enable when the corpus has more variants-table documents to amortize the maintenance cost. See TODO #84."*

So the **current production behavior** for tables, on both sides of the pipeline, is:

- **Graph extraction side:** raw flattened table text appears in `docling_document_json["texts"]` for each table (Docling-generated). `extract_table_overlay()` runs at `main.py:1185` to produce the Phase 0/0.5 overlay (consumed at merge time by `app/services/table_overlay.py`). `synthesize_table_facts()` is **not called** in production — only in tests. `run_extraction_pass` consumes raw `texts[]` as input.
- **Embedding side:** `app/services/chunking.py:113-126` emits one opaque `modality="table"` chunk per table element, with `text` = Docling's flattened content, regardless of size. For tables exceeding the `BAAI/bge-large-en-v1.5` hard limit of 512 tokens, the embedding is silently truncated to the first ~512 tokens.

### 1.2 What's wrong with that

For wide spec-sheet tables like the SA-2 variants table (~2,000–3,000 tokens of flattened content):

- **Embedding side:** the chunk is silently truncated, so variant-specific queries (*"S-75M2 max range"*) often don't match.
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
- `docker/docling-graph/app/_table_facts.py`, `_alias_map.py` — stay on disk. Not in production today (per §1.1). New `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` flag (default `false`) controls whether they run at all.
- `docker/docling-graph/tests/test_table_facts_*.py` — stay green; they test the experimental path directly via module imports, independent of the production-flag wiring.
- `app/services/table_overlay.py` — untouched. Phase 0/0.5 merge-time machinery is orthogonal: it consumes the overlay produced by `extract_table_overlay()` (which reads `body.docling_document_json` directly, not `texts[]`), and the overlay machinery does not depend on flat `texts[]` representation of tables. Verified at `main.py:1185-1228`.

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
    raw_markdown: str                # captured for OTHER fallback + small-table whole rendering. SOURCE: the matching `#/texts/N` element whose ref points to this table, or `doc_json["tables"][i]["text"]` if no flat-text mirror is present.
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

7. **`raw_markdown` capture.** For every table including OTHER, capture the Docling-generated flat text representation. **Source resolution (explicit):**
   - First preference: locate the `#/texts/N` element whose ref points to this table (Docling generally generates a flat-text mirror of each table); use that element's `.text`.
   - Fallback: `doc_json["tables"][i]["text"]` if present.
   - Final fallback: `doc_json["tables"][i].get("data", {}).get("table_markdown", "")` or empty string. Logged at DEBUG when fallback fires.

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
    token_limit_whole: int = 1500,
    token_limit_column: int = 1200,
) -> list[GraphTableChunk]: ...
```

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

### 9.2 Invariant: `_suppress_raw_table_texts` does NOT mutate `doc_json["tables"]`

```python
def _suppress_raw_table_texts(doc_json: dict, normalized: list[NormalizedTable]) -> None:
    """Remove flat-text mirrors of normalized tables from doc_json['texts'].

    Invariants (asserted by unit test):
    - doc_json['tables'] is NOT touched. The Phase 0/0.5 overlay machinery
      (extract_table_overlay → table_overlay.py merge-time application) reads
      from doc_json['tables'] directly and must remain functional.
    - Tables with shape == OTHER keep their flat text in texts[] — the
      embedding-side OTHER fallback and the graph-side OTHER fallback both
      depend on the original text remaining visible.
    - Only entries in texts[] whose `self_ref` matches `#/tables/{i}` for
      i in [nt.table_index for nt in normalized if nt.shape != OTHER]
      are removed.
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    target_prefixes = tuple(f"#/tables/{i}" for i in non_other)
    doc_json["texts"] = [
        t for t in doc_json["texts"]
        if not t.get("self_ref", "").startswith(target_prefixes)
    ]
```

## 10. Embedding Renderer (`render_embedding.py`)

```python
def render_for_embedding(
    table: NormalizedTable,
    token_limit: int = 512,                 # from existing app/config.py:400 (embedding_chunk_max_tokens)
    summary_limit: int = 300,               # from new EMBEDDING_TABLE_SUMMARY_MAX_TOKENS
) -> list[EmbeddingTableChunk]: ...
```

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

### 10.1 Integration in `app/services/chunking.py`

```python
def is_table_normalization_enabled_embedding() -> bool:
    return os.environ.get("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "true").lower() != "false"

# In structure_aware_chunk(...):
elif etype == "table":
    current_heading = buffer_heading
    _flush_buffer()
    if not is_table_normalization_enabled_embedding():
        # Master kill-switch: today's behavior (single opaque chunk).
        chunks.append(StructuredChunk(text=content, chunk_index=chunk_index,
                                      modality="table", ...))
        chunk_index += 1
    else:
        nt = normalized_table_for(elem, normalized_tables)
        if nt is None or nt.shape == Shape.OTHER:
            chunks.append(StructuredChunk(text=content, chunk_index=chunk_index,
                                          modality="table", ...))
            chunk_index += 1
        else:
            for etc in render_for_embedding(nt):
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

`StructuredChunk` gains optional `metadata: dict | None = None`. `normalized_tables` is computed once per document (caller-supplied; embedding ingest path invokes `normalize_tables(doc_json)` and threads the result down).

**Master kill-switch correctness:** with `EMBEDDING_TABLE_NORMALIZATION_ENABLED=false`, the only difference vs today's code is the added `if` check at the top — chunk emission falls through to the *exact same* `StructuredChunk(...)` call. Asserted by the byte-equality test in §14.

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

### 11.4 ArcadeDB mirroring (--blow-away mandatory)

`TextChunk` vertices in ArcadeDB gain one new property: `chunk_kind: str | None`, populated from `chunk_metadata.chunk_kind`. **This property cannot be added in-place to an existing populated vertex type without rebuilding the schema.** The user's `./manage.sh --blow-away` flow handles this; any deployment path that skips blow-away will fail at ArcadeDB write time with a schema mismatch. Documented in the rollout section (§13) and added to `VERIFICATION_CHECKLIST.md` (§14).

`arcadedb_schema.py` gains a new index declaration for `chunk_kind` on `TextChunk`. `arcadedb_client.py` / `arcadedb_graph.py` write path includes the property when present.

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

`GraphTableChunk.cell_refs` flow into the existing extraction-provenance pipeline. `_text_item_from_chunk(gtc)` (converts a chunk into a docling `TextItem` for `texts[]`) attaches `gtc.cell_refs` to the chunk-trace map that the existing field-provenance walker (`app/services/extraction_merge.py`, `docker/docling-graph/app/_field_provenance_helpers.py`) reads.

Result: extracted `Missile.max_range_m` on `S-75M2` → `ExtractionFieldProvenance.evidence_ids` contains *both* the graph-chunk self_ref AND the underlying cell ref. Two-hop provenance, both traceable.

No code changes to the field-provenance walker — cell refs appear as additional entries in `evidence_ids`. (Provenance plumbing is fresh — recent commits added this surface; spot-check during implementation.)

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

1. **Step 0 (BEFORE any code changes)** — capture baseline. See §19.
2. Apply alembic migration (additive nullable column + partial index).
3. Code merges to `main` with all master switches at `false` defaults. Production behavior is byte-identical to pre-merge.
4. Run integration test `test_master_kill_switch_byte_equality.py` against the SA-2 corpus to confirm `doc_json["texts"]` and chunk outputs are unchanged vs. baseline. **This is a merge gate.**

### Phase 2 — Behavior changes, gated on regression check

5. Capture current-production extraction outputs on SA-2 corpus (call this the "today baseline" — distinct from the §19 fixture, which is `texts[]`/chunk-level, while this is extraction-level).
6. Flip `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true` + `EMBEDDING_TABLE_NORMALIZATION_ENABLED=true`. Run `./manage.sh --blow-away` and re-ingest.
7. Score extraction against the today-baseline.
8. **Merge gate (§15.1) on the flag flip itself, not on the code merge:** new path must produce **no regression vs today-baseline** on the SA-2 `missile_propulsion` ✓ exact count. Strict criterion: new ✓ exact ≥ today-baseline ✓ exact AND no other pass regresses by more than 1 ✓ exact (each pass independently — i.e., for every individual pass, new ≥ today − 1).
9. If criterion fails: flip flags back to `false` (cost: another `--blow-away` + re-ingest). File a follow-up issue with the captured comparison data.

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
| Master kill-switch byte equality | With both `*_NORMALIZATION_ENABLED=false`, `doc_json["texts"]` and embedding chunks on SA-2 are byte-identical to the §19 baseline fixture | `tests/integration/test_master_kill_switch_byte_equality.py` |
| Storage round-trip | Write/read `chunk_metadata`; partial index returns matching rows | `tests/integration/test_chunk_metadata_persistence.py` |
| Retrieval surfacing | `/v1/retrieval/query` returns `table_chunk` block when `chunk_metadata` non-null; absent otherwise | `tests/integration/test_retrieval_table_chunk_surfacing.py` |
| Graph provenance | `ExtractionFieldProvenance.evidence_ids` includes underlying cell refs for SA-2 propulsion extraction | `tests/integration/test_graph_provenance_cell_refs.py` |
| End-to-end SA-2 | Ingest → extract → score → cell_refs traceable | `tests/integration/test_sa2_table_pipeline_e2e.py` |
| Experimental-path drift guard | `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` produces the same `synthesize_table_facts()` output the existing `test_table_facts_*.py` tests expect. Specific assertion: count of `TextItem`s emitted matches an empirically-determined fixture for SA-2; first 3 emitted items match snapshot. **This guards against `_table_facts.py` rotting on disk; it is NOT a baseline-matching test.** | `tests/integration/test_experimental_table_facts_drift.py` |

**Additions to `VERIFICATION_CHECKLIST.md`:**
- `□ Step 0 baseline fixture committed to tests/fixtures/sa2_today_baseline.json (commit SHA recorded)`
- `□ Phase 1 merge: master kill-switch byte-equality test passes`
- `□ Phase 2 flip: SA-2 missile_propulsion ✓ exact ≥ today-baseline ✓ exact`
- `□ Phase 2 flip: no other pass regresses by >1 ✓ exact`
- `□ ArcadeDB chunk_kind property requires --blow-away (documented)`
- `□ .env + .env.example contain all 7 new variables`

## 15. Success Criteria (acceptance gates)

### 15.1 Phase 1 merge gate (code-merge time)

- All unit tests pass.
- `test_master_kill_switch_byte_equality.py` proves both sides emit byte-identical chunks vs. the §19 baseline with master switches off.
- `test_suppress_raw_table_texts_invariant.py` proves the `tables[]`-untouched invariant.
- `test_render_column_byte_equality.py` proves both renderers share format.

### 15.2 Phase 2 flip gate (production-flip time)

- SA-2 `missile_propulsion` ✓ exact ≥ today-baseline ✓ exact (no regression).
- For every other pass: new ✓ exact ≥ today-baseline ✓ exact − 1 (≤1-count regression allowed per individual pass, not summed).
- `/v1/retrieval/query` for *"S-75M2 max range"* returns top-1 result with populated `table_chunk` block and correct `cell_refs`.
- Extracted `Missile.max_range_m` on `S-75M2` has `ExtractionFieldProvenance.evidence_ids` containing both graph-chunk self_ref AND underlying cell ref.

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
1. On current `main`, ingest the SA-2 corpus.
2. For each SA-2 document, capture and commit under `tests/fixtures/`:
   - **`sa2_<docid>_texts_today.json`** — full `doc_json["texts"]` after sanitization but before `run_extraction_pass`. This is the chunk-level baseline for `test_master_kill_switch_byte_equality.py` (§15.1).
   - **`sa2_<docid>_extraction_today.json`** — the full pass-result outputs across all passes for that document. This is the extraction-level baseline for the Phase 2 flip gate (§15.2).
3. Record in `tests/fixtures/sa2_today_baseline.meta.json`:
   ```json
   {
     "captured_at": "2026-05-11T...",
     "main_sha": "<git rev-parse HEAD>",
     "docling_graph_image_sha": "<docker compose images --format json>",
     "corpus_files": ["...", "..."],
     "extraction_pass_counts": {"<doc_id>": {"missile_propulsion": {"exact": N, "wrong": N, "null": N}}, ...}
   }
   ```
4. Commit the fixtures on `feat/table-aware-chunking` with message `test(baseline): capture today's SA-2 production behavior pre-rewrite (sha <SHA>)`.

### Why this matters

Without this, the merge gate at §15.1 and the flip gate at §15.2 are aspirational. With it, both gates have concrete, version-controlled targets to assert against. The fixture also doubles as the regression target for any future change to the table-chunking layer.

### Captured artifacts are stable

The procedure captures *deterministic* outputs only:
- `doc_json["texts"]` after sanitization — pure function of input doc + sanitizer, no LLM call.
- Extraction outputs — captured with `temperature=0` runs (or whichever temperature the user's `think_true_*.csv` runs use, recorded in the meta file) to keep variance low.

If LLM-output variance is too high to make ✓ exact comparison meaningful even at `T=0`, the comparison should be median-of-N runs (N=3) with the threshold relaxed by 1.
