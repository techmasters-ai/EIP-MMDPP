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

### 10.1 Integration in `app/services/chunking.py`

```python
def is_table_normalization_enabled_embedding() -> bool:
    # Default "false" matches §12 (master switches ship disabled).
    return os.environ.get("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "false").lower() == "true"

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

### 11.4 ArcadeDB mirroring

`TextChunk` vertices in ArcadeDB gain one new property: `chunk_kind: str | None`, populated from `chunk_metadata.chunk_kind`. `arcadedb_schema.py` gains a new index declaration for `chunk_kind` on `TextChunk`. `arcadedb_client.py` / `arcadedb_graph.py` write path includes the property when present.

**Migration posture:** ArcadeDB supports `CREATE PROPERTY ... IF NOT EXISTS` and `CREATE INDEX ... IF NOT EXISTS` for in-place schema additions on existing populated vertex types. The schema-bootstrap path in `arcadedb_schema.py` should add the property + index idempotently on startup. **However**, since the user's rollout flow (§13) uses `./manage.sh --blow-away` followed by re-ingest, the in-place migration path is not exercised in this rollout. Still, the schema-bootstrap code should be written defensively (idempotent property/index creation) so future fresh deployments behave correctly. **Implementation must verify** by reading `arcadedb_schema.py` patterns for existing properties — follow the convention used there rather than inventing a new approach (per `feedback_prefer_native_libraries`).

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

**Expectation:** no code changes to the field-provenance walker — cell refs appear as additional entries in `evidence_ids` simply by being present in the chunk-trace map.

**If this expectation does not hold during implementation** (e.g., the walker filters refs by prefix and rejects `#/tables/...` pointers, or the chunk-trace map's value type can't carry the new refs), this is a **blocker**, not a workaround opportunity. Stop, file a follow-up issue, and either (a) add a single targeted change to the walker with its own test, or (b) deferred provenance-wiring to a follow-up PR and ship the rest of the spec — but do not silently drop `cell_refs` on the floor. The integration test `test_graph_provenance_cell_refs.py` is the gate that detects this.

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
| Master kill-switch byte equality | With both `*_NORMALIZATION_ENABLED=false`, `doc_json["texts"]` and embedding chunks on SA-2 are byte-identical to the §19 baseline fixture | `tests/integration/test_master_kill_switch_byte_equality.py` |
| Storage round-trip | Write/read `chunk_metadata`; partial index returns matching rows | `tests/integration/test_chunk_metadata_persistence.py` |
| Retrieval surfacing | `/v1/retrieval/query` returns `table_chunk` block when `chunk_metadata` non-null; absent otherwise | `tests/integration/test_retrieval_table_chunk_surfacing.py` |
| Graph provenance | `ExtractionFieldProvenance.evidence_ids` includes underlying cell refs for SA-2 propulsion extraction | `tests/integration/test_graph_provenance_cell_refs.py` |
| End-to-end SA-2 | Ingest → extract → score → cell_refs traceable | `tests/integration/test_sa2_table_pipeline_e2e.py` |
| Experimental-path drift guard | `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` produces non-empty `synthesize_table_facts()` output on SA-2 fixture. **Assertion: count of `TextItem`s emitted > 0 AND `TextItem` count is within ±10% of a recorded fixture value.** Count-based, not text-content snapshot — tolerates whitespace/cosmetic drift but catches catastrophic rot (the path stops emitting anything, or emits 10× too much). **This guards against `_table_facts.py` rotting on disk; it is NOT a baseline-matching test.** | `tests/integration/test_experimental_table_facts_drift.py` |
| Disallowed-combination fallback | Both `*_ENABLED` master flags AND `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true` simultaneously → falls back to today's raw-blob behavior (per §9.1 row 4). Asserted: `texts[]` byte-identical to §19 baseline; ERROR log emitted. | `tests/integration/test_disallowed_combination_fallback.py` |

**Additions to `VERIFICATION_CHECKLIST.md`:**
- `□ Step 0 baseline fixtures committed under tests/fixtures/sa2/ (main_sha recorded in baseline.meta.json)`
- `□ Phase 1 merge: master kill-switch byte-equality test passes against §19 baseline`
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
- For every other pass: new ✓ exact ≥ today-baseline ✓ exact − 1 (≤1-count regression allowed per individual pass, not summed).
- **Corpus-wide guard:** sum of ✓ exact across all passes ≥ sum of today-baseline ✓ exact − 2 (prevents the "every pass loses 1" scenario where each individual pass passes the per-pass gate but corpus-level quality silently degrades).

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
1. On current `main`, ingest the SA-2 corpus.
2. **Run the corpus N=3 times** (the LLM variance check determines whether the §15.2 gate compares against best-of-3, median-of-3, or strict single-run; see decision rule below).
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
