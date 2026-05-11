# Table-Aware Chunking — Design

**Status:** Draft 2026-05-11
**Branch:** `feat/table-aware-chunking`
**Related (preserved as dormant fallback):**
- `2026-05-05-section-aware-table-fact-synthesis-design.md`
- `2026-05-06-table-identity-rewrite-and-field-overlay-design.md`

---

## 1. Problem

The codebase currently handles tables differently on each side of the pipeline,
and the embedding side has a gap that hurts retrieval quality for wide
spec-sheet tables.

**Graph extraction side (today):** `docker/docling-graph/app/_table_facts.py`
(1,468 LOC, approved 2026-05-05) parses Docling table_cells and emits one
`TextItem` per `(entity, schema_field, value)` triple for the LLM. Row labels
are pre-resolved to schema fields via `_alias_map.py` *before* the LLM sees
the prompt. This was built specifically to fix the SA-2 propulsion column-
arithmetic failure (✓0 → ✓≥6 at T=1.0). It works.

**Embedding side (today):** `app/services/chunking.py:113–126` treats every
table element as a single opaque chunk whose `text` is whatever Docling
flattened. The current embedding model is `BAAI/bge-large-en-v1.5` with a
hard limit of 512 tokens; the chunker's `DEFAULT_MAX_CHUNK_TOKENS = 512`
matches that, but the table path emits the chunk regardless of size. For a
wide table like the SA-2 spec sheet (~2,000–3,000 tokens of flattened
content), the embedding is silently truncated to the first ~512 tokens —
queries like *"S-75M2 max range"* may never match because that variant's
column never makes it into the embedded representation.

**What we want:** a table-aware chunking layer that emits structured,
self-contained chunks tuned for each consumer — preserving retrieval
precision on the embedding side and providing the graph LLM with chunks it
can extract from without losing context to a 4k-token table blob.

The design is informed by an external analysis (provided 2026-05-11) that
proposed a shared normalization layer feeding two renderers. This spec
adapts that approach to fit the existing pipeline and explicitly preserves
the empirical SA-2 propulsion fix behind a dormant fallback flag.

## 2. Goals

1. **Embedding-side retrieval precision.** Variant-specific queries
   (e.g., *"S-75M2 max range"*) reliably retrieve the correct per-variant
   chunk for wide tables that today are silently truncated.
2. **Graph-side self-contained chunks.** Replace the raw flattened table
   text (the active source of column-arithmetic confusion) with per-entity
   column chunks formatted to preserve identity + section + spec rows
   together.
3. **Provenance traceability.** Every emitted chunk carries `cell_refs`
   pointing back to specific `#/tables/N/data/table_cells/M` entries so
   retrieval results and extracted fields can be traced to source cells.
4. **Regression safety.** The empirically-validated SA-2 propulsion fix
   path (`_table_facts.py` + `_alias_map.py`) remains on disk, dormant,
   gated behind a single env flag. One flag flip restores today's
   validated behavior without redeploy.
5. **Master kill-switch.** When normalization is disabled, behavior is
   identical to today on both sides.

## 3. Non-Goals (explicit out-of-scope)

- Multilingual row labels (deferred — same as D5 in the prior spec).
- Prose-table hybrids (tables embedded in flowing prose without proper
  Docling parsing).
- Cross-table consolidation (e.g., merging two SA-2 tables that share
  variants).
- Notebook outcome-tracker `facts/pass` column updates.
- Retrieval-side reranking boosts by `chunk_kind` (a follow-up enabled
  by the indexed column added here).
- UI surfacing of the `table_chunk` block in the document viewer.
- Generated `§12b` prose from normalized tables.
- Auto-extending the section-keyword list or the spec-row keyword set —
  these stay hand-coded; additions are one-line changes + a test.
- A backward-compatibility re-chunk script — rollout uses
  `./manage.sh --blow-away` + re-ingest, per user decision.

## 4. Architecture Overview

```
docling_document_json["tables"]
            │
            ▼
   ┌────────────────────┐
   │  normalize_tables  │  ← single pass; renderer-agnostic
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

The `NormalizedTable` model is the only contract between normalization
and rendering. Both renderers are pure functions; they share an internal
helper (`_render_column_as_text`) that produces the identity+sections+rows
block, so the chunk format is identical across consumers.

## 5. Module Layout

```
app/services/table_normalization/
├── __init__.py                 # public API: normalize_tables(), render_for_graph(), render_for_embedding()
├── models.py                   # NormalizedTable, NormalizedRow, NormalizedColumn, NormalizedCell, TableSection, ChunkKind
├── detect.py                   # shape detection: column_major | row_major | hybrid | other
├── normalize.py                # docling table_cells → NormalizedTable
├── render_graph.py             # NormalizedTable → list[GraphTableChunk]
├── render_embedding.py         # NormalizedTable → list[EmbeddingTableChunk]
├── tokens.py                   # bge-m3 tokenizer-aware size check (shared)
└── config.py                   # env-flag reading + default thresholds
```

**Cross-image distribution:** `app/services/table_normalization/` is the
single source of truth. `docker/docling-graph/Dockerfile` adds a `COPY`
directive that mirrors the directory into `/app/app/services/table_normalization`
inside the docling-graph image at build time. The api/worker/worker-graph
services use the bind-mounted `./app` and pick up changes without rebuild.
This pattern is consistent with `feedback_container_rebuild` — docling-graph
needs `docker compose build docling-graph`.

**What lives elsewhere (unchanged):**
- `docker/docling-graph/app/_table_facts.py` — stays on disk, gated off by default.
- `docker/docling-graph/app/_alias_map.py` — stays on disk, used only when legacy flag flipped.
- `docker/docling-graph/app/_table_pivot.py` — **deleted** (already deprecated, no consumers when legacy flag is the new fallback).
- `app/services/table_overlay.py` — untouched. Phase 0/0.5 merge-time machinery operates on overlays emitted in LLM responses; orthogonal to chunk emission.

## 6. Data Model (`models.py`)

```python
class Shape(str, Enum):
    COLUMN_MAJOR = "column_major"
    ROW_MAJOR    = "row_major"
    HYBRID       = "hybrid"          # column-major + multiple identity rows
    OTHER        = "other"           # skip; fall back to raw rendering


@dataclass(frozen=True)
class CellRef:
    """JSON-Pointer back into docling_document_json."""
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
    identity: dict[str, str]         # all identity-row values for this column
    display_name: str                # heuristic: prefer Industry → Military → NATO → Missile Type; fallback "col-{n}"


@dataclass(frozen=True)
class TableSection:
    name: str
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class NormalizedTable:
    table_index: int
    self_ref: str                    # "#/tables/3"
    caption: str | None
    page_numbers: tuple[int, ...]
    shape: Shape
    rows: tuple[NormalizedRow, ...]
    columns: tuple[NormalizedColumn, ...]
    sections: tuple[TableSection, ...]
    cells: tuple[NormalizedCell, ...]
    raw_markdown: str                # captured for OTHER fallback + small-table whole rendering
```

All `frozen=True`. The model is immutable post-construction. Row-major tables
are represented identically — `NormalizedColumn` corresponds to a *row* in
the source, with its row-header text as the identity. From the renderer's
perspective, "entity = column" is uniform.

## 7. Shape Detection (`detect.py`)

Pure function: `detect_shape(table_cells, table_data) -> Shape`. Operates
only on Docling-provided signals (`row_header`, `column_header`,
`start_row_offset_idx`, `end_row_offset_idx`, `start_col_offset_idx`,
`end_col_offset_idx`, `text`).

**Detection rules, in order:**

1. **Floor:** `num_rows < 4` or `num_cols < 4` → `OTHER`. Below this size the
   column-arithmetic failure mode does not apply.

2. **`COLUMN_MAJOR` test:** ≥50% of non-empty `start_col_offset_idx == 0`
   cells have `row_header: True`, AND at least one row label matches a
   builtin spec-row keyword (`max range`, `weight`, `length`, `diameter`,
   `altitude`, `velocity`, `speed`, `mass`, `power`, `frequency`, etc.).

3. **`HYBRID` upgrade:** Applied after `COLUMN_MAJOR`. Count rows 0..N where
   every data cell is identity-shaped (short text < 40 chars, non-numeric,
   non-blank). If ≥2 such rows exist at the *top* → upgrade to `HYBRID`.
   SA-2 has 4 identity rows (Industry / Military / NATO / Fan Song).

4. **`ROW_MAJOR` test:** ≥50% of non-empty row-0 cells have
   `column_header: True`, AND row-0 contains spec keywords.

5. **Section-header detection (all non-OTHER shapes):** A row is a section
   header when all non-empty cells span the full data-column width
   (`col_span ≥ num_data_cols`) AND cell text matches a builtin section
   keyword (`1st stage`, `2nd stage`, `booster`, `sustainer`, `missile`,
   `radar`, `launcher`, `propulsion`, `guidance`, plus ~5 more).

6. **Fallback:** Anything that doesn't match → `OTHER`. Per-table failures
   never propagate; `normalize_tables` emits a `NormalizedTable(shape=OTHER,
   cells=(), raw_markdown=...)` so weird tables survive as one chunk.

**Diagnostics emitted alongside Shape (flows into pass diagnostics):**
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

**Explicit non-features:**
- No LLM-assisted shape guessing — deterministic, testable heuristic only.
- No alias-map dependency in detection — shape is independent of which
  schema fields any pass cares about. Same shape for every pass on every doc.
- No partial-shape handling (e.g., "first half column-major, second half
  row-major"). Will revisit when such a table is observed in the wild.

## 8. Normalization (`normalize.py`)

Entry point:

```python
def normalize_tables(doc_json: dict) -> list[NormalizedTable]:
    """Pure function. Reads doc_json['tables']; never writes doc_json.
    Returns one entry per table including OTHER (with empty cells, raw_markdown set)."""
```

**Per-table pipeline:**

1. **Shape detection** via `detect.py`. `OTHER` → emit minimal `NormalizedTable`,
   skip steps 2–7.

2. **Build `NormalizedRow` list.**
   - Column-major/hybrid: iterate rows; row label = text of cell at
     `start_col_offset_idx == 0` (or merged label cell spanning col 0).
     Identity rows flagged per step 3 of detection. Section-header rows
     flagged per step 5. Unit extracted from label suffix via regex
     `r"\(\s*([a-zA-Z/°²³]+)\s*\)\s*$"` — matches `(m)`, `(mm)`, `(m/s)`,
     `(°)`, etc.
   - Row-major: conceptually transposed; "row label" comes from column
     header in row 0; same identity/section/spec logic applied.

3. **Build `NormalizedColumn` list (entities).**
   - Column-major/hybrid: each non-label column is one entity. Identity
     dict built by walking every identity row and reading the cell at
     that column. Empty identity cells fall back via `colspan` propagation
     (Docling sometimes fills only the first column of a span).
   - `display_name` heuristic: prefer `Industry Designation`, then
     `Military Designation`, then `NATO Designation`, then `Missile Type`.
     If none present → `"col-{col_idx}"` stable fallback.
   - Row-major: each non-header row becomes a `NormalizedColumn`.

4. **Section assignment.** Walk row list top-to-bottom; section-header
   rows reset the section context for all subsequent spec rows until the
   next section header. Rows above the first section header have
   `section = None`. `TableSection` entries emitted in document order.

5. **Build `NormalizedCell` list.** For each (spec row × entity column)
   pair, emit one cell with: `value` = raw text (strip whitespace; no
   numeric coercion — preserves multi-value cells like `"1135/1028"`),
   `unit` = row's inherited unit (or `None`), `column_identity` = full
   identity dict, `cell_ref` populated. **Empty cells are skipped.**

6. **Merged-cell handling.**
   - Column span (`start_col_offset_idx != end_col_offset_idx`):
     value replicated across each spanned column.
   - Row span (`start_row_offset_idx != end_row_offset_idx`): value
     replicated down. Rare in spec tables; supported for completeness.

7. **`raw_markdown` capture.** For every table including OTHER, the
   existing pipeline's flattened-text representation is stored on
   `NormalizedTable.raw_markdown`. Used as: OTHER fallback path,
   embedding-side whole-table chunk source when table fits, and input
   to the legacy `_table_facts.py` path when the dormant flag is flipped.

**Error handling.** Any per-table exception caught, logged at WARNING with
`table_index` + `self_ref`, produces `NormalizedTable(shape=OTHER, cells=(),
raw_markdown=...)` for that table. Other tables continue normally. **No
table failure breaks the pipeline.** Matches `_table_facts.py`'s safety
posture and the user's data-lineage requirement.

**Idempotency.** `normalize_tables(doc_json)` reads but never writes
`doc_json`. Callers cache the result once per pass — no internal caching.

**Diagnostics emitted (per call, aggregated across all tables):**
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
}
```

Routed into `diagnostics["service_table_normalization"]`.

## 9. Graph Renderer (`render_graph.py`)

```python
def render_for_graph(
    table: NormalizedTable,
    token_limit_whole: int = 1500,   # from DOCLING_GRAPH_TABLE_WHOLE_LIMIT
    token_limit_column: int = 1200,  # from DOCLING_GRAPH_TABLE_COLUMN_LIMIT
) -> list[GraphTableChunk]: ...
```

```python
@dataclass(frozen=True)
class GraphTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: str                  # "table_whole" | "table_entity_column" | "table_entity_section"
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]
```

**Decision tree:**

1. Shape == OTHER → one `table_whole` chunk with `text = raw_markdown`,
   empty cell_refs. Same behavior as today; no table disappears.

2. Whole-table rendering ≤ `token_limit_whole` → one structured `table_whole`
   chunk (identity rows + sections + spec rows in one block).

3. Whole-table > `token_limit_whole` → per-column emission:
   - For each `NormalizedColumn`: render as one chunk.
   - If single-column rendering > `token_limit_column`: split by `TableSection`,
     each section chunk repeating the column's identity header
     (`chunk_kind = "table_entity_section"`).

**Chunk text format (per the analysis):**

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
- <row_label>: <value> <unit>

<SECTION 2 NAME>:
- <row_label>: <value> <unit>
```

- Section names uppercased (`MISSILE`, `1ST STAGE`).
- Rows without a section appear under `GENERAL:`.
- Units come from row's inherited unit; absent units render value alone.
- Multi-value cells (`"1135/1028"`) rendered as-is.
- Empty values skipped (handled in `normalize.py`).
- Raw markdown of the source table is **not** included.

**Token measurement** uses `tokens.py` (BAAI/bge-m3 tokenizer loaded
lazily once per process). Same tokenizer the embedding pipeline already
uses (`embedding_chunk_tokenizer_model` in `app/config.py:402`).

**Integration in `docker/docling-graph/app/main.py`:**

```python
normalized = normalize_tables(doc_json)

if os.environ.get("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true").lower() != "false":
    if os.environ.get("DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS", "false").lower() != "true":
        # New path (default).
        for nt in normalized:
            for gtc in render_for_graph(nt):
                doc_json["texts"].append(_text_item_from_chunk(gtc))
        if os.environ.get("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", "true").lower() != "false":
            _suppress_raw_table_texts(doc_json, normalized)
    else:
        # Legacy A/B fallback.
        synthesize_table_facts(doc_json, ...)
else:
    # Master kill-switch: preserve current behavior including raw text + legacy path.
    synthesize_table_facts(doc_json, ...)
```

`_suppress_raw_table_texts` removes `texts[]` entries whose `self_ref`
starts with `#/tables/{i}` for each non-OTHER normalized table. Tables
with shape OTHER keep their raw text.

## 10. Embedding Renderer (`render_embedding.py`)

```python
def render_for_embedding(
    table: NormalizedTable,
    token_limit: int = 512,          # from EMBEDDING_CHUNK_MAX_TOKENS
) -> list[EmbeddingTableChunk]: ...
```

```python
@dataclass(frozen=True)
class EmbeddingTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: str                  # "table_summary" | "table_whole" | "table_entity_column" | "table_entity_section"
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]
```

**Emission rules:**

1. Shape == OTHER → one `table_whole` chunk with `text = raw_markdown`.
   Same behavior as today.

2. **Always emit `table_summary`** (one per table; capped at
   `EMBEDDING_TABLE_SUMMARY_MAX_TOKENS = 300`):

    ```
    TABLE: <caption>
    SOURCE: page <pages>; ref <table_ref>
    VARIANTS: <display_names; truncated with "..." if too many>
    PROPERTIES: <spec-row labels; identity/section excluded; truncated similarly>
    ```

3. **Whole-table rendering ≤ `token_limit`** → also emit `table_whole` chunk
   (same structured format as graph renderer's whole-table path).

4. **Whole-table > `token_limit`** → emit per-entity-column chunks (same
   shape as graph renderer). If a single column > `token_limit`: split by
   section, identity header repeated → `table_entity_section`. The
   `table_whole` chunk is **not** emitted in this branch.

**Shared rendering helper:** `_render_column_as_text(column, table, sections)`
produces the identity+sections+rows block. Both renderers call it directly.
One source of truth for chunk format.

**Integration in `app/services/chunking.py`** — the `elif etype == "table":`
branch (lines 113–126) becomes:

```python
elif etype == "table":
    current_heading = buffer_heading
    _flush_buffer()
    if not is_table_normalization_enabled():
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
                        "chunk_kind": etc.chunk_kind,
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

`StructuredChunk` gains an optional `metadata: dict | None = None` field.
The persistence layer writes it to the new `chunk_metadata` JSONB column.
`normalized_tables` is computed once per document (caller-supplied;
typically the embedding ingest path invokes `normalize_tables(doc_json)`
and threads the result down).

## 11. Provenance & Storage

### 11.1 Alembic migration (`alembic/versions/00XX_chunk_metadata.py`)

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
    op.drop_index("ix_text_chunks_chunk_kind", table_name="text_chunks", schema="retrieval")
    op.drop_column("text_chunks", "chunk_metadata", schema="retrieval")
```

Additive; nullable column. Partial expression index on `chunk_kind` is cheap
(most chunks are `chunk_metadata IS NULL`) and prepares for future
retrieval-side filtering / scoring by chunk kind.

### 11.2 `chunk_metadata` payload schema

For normalized table chunks:

```json
{
  "chunk_kind": "table_entity_column",
  "table_ref": "#/tables/3",
  "entity_display_name": "S-75M2 / SA-2D",
  "section": null,
  "column_index": 7,
  "cell_refs": [
    "#/tables/3/data/table_cells/42",
    "#/tables/3/data/table_cells/43",
    "#/tables/3/data/table_cells/47"
  ],
  "row_labels": ["Max Range", "Min Range", "Max Altitude"],
  "table_caption": "S-75 Technical Data / SA-2 Guideline Variant Specifications",
  "page_numbers": [6, 7]
}
```

For non-table chunks (prose, code, heading, equation, image-caption):
`chunk_metadata` stays `NULL`.

### 11.3 Model update (`app/models/retrieval.py`)

```python
class TextChunk(Base, TimestampMixin):
    # ... existing fields ...
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
```

Only model change.

### 11.4 ArcadeDB mirroring

`TextChunk` vertices in ArcadeDB carry the vector + chunk text today. They
gain one new property: `chunk_kind: str | None`, populated from
`chunk_metadata.chunk_kind`. Single indexed string property only — full
`chunk_metadata` is NOT mirrored to ArcadeDB; Postgres remains the source
of truth for structured metadata.

`arcadedb_schema.py` gets a new index declaration for `chunk_kind`.
`arcadedb_client.py` / `arcadedb_graph.py` write path includes the new
property when present.

### 11.5 Retrieval surfacing (`/v1/retrieval/query`)

The existing response shape (per the recent commit `feat(retrieval):
surface evidence_ids/self_refs/page_numbers`) is extended with an optional
`table_chunk` block per result when `chunk_metadata IS NOT NULL`:

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

Additive field. Backwards-compatible with consumers that don't know about it.

### 11.6 Graph-side provenance wiring

`GraphTableChunk.cell_refs` flow into the existing extraction-provenance
pipeline. The helper `_text_item_from_chunk(gtc)` (converts a chunk into a
docling `TextItem` for `texts[]`) attaches `gtc.cell_refs` to the same
chunk-trace map that the existing field-provenance walker reads from.

Result: when an extracted `Missile` entity for `S-75M2` reports
`max_range_m = 56000`, its `ExtractionFieldProvenance.evidence_ids`
contains *both* the graph-chunk self_ref AND the underlying cell ref
`#/tables/3/data/table_cells/42`. Two-hop provenance, both traceable.

No changes to the field-provenance walker; cell refs simply appear as
additional entries in `evidence_ids`.

## 12. Configuration

All variables land in both `.env` and `.env.example` with default value
and one-line comment, per `feedback_env_vars_must_appear_in_dotenv_files`.

| Variable | Default | Side | Effect |
|---|---|---|---|
| `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED` | `true` | Graph | Master kill-switch (false → today's behavior). |
| `DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS` | `false` | Graph | A/B fallback (true → run dormant `_table_facts.py`). |
| `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN` | `true` | Graph | True → strip raw flattened table text after normalization. |
| `DOCLING_GRAPH_TABLE_WHOLE_LIMIT` | `1500` | Graph | Whole-vs-column threshold (tokens). |
| `DOCLING_GRAPH_TABLE_COLUMN_LIMIT` | `1200` | Graph | Split-column-by-section threshold (tokens). |
| `EMBEDDING_TABLE_NORMALIZATION_ENABLED` | `true` | Embedding | Master kill-switch for embedding side. |
| `EMBEDDING_CHUNK_MAX_TOKENS` | `512` | Embedding | Whole-vs-column threshold (matches `bge-large-en-v1.5` limit). |
| `EMBEDDING_TABLE_SUMMARY_MAX_TOKENS` | `300` | Embedding | Summary-chunk cap. |

Reads centralized in `app/services/table_normalization/config.py` so flag
names and defaults appear in exactly one source file.

## 13. Rollout

Single PR. Migration applies first (additive nullable column + partial
expression index). Code lands on `feat/table-aware-chunking` and merges to
`main`. User runs `./manage.sh --blow-away` and re-ingests. No phased
rollout, no opt-in re-ingest script.

**A/B procedure:**
- **Default run:** all flags at defaults (new path active, raw suppressed).
  Re-ingest SA-2 → run extraction → score against GT.
- **Fallback run:** `DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS=true`, re-ingest →
  extract → score. Validated baseline (✓ exact ≥ 6 on `missile_propulsion`
  at T=1.0).
- **Embedding-only A/B:** flip `EMBEDDING_TABLE_NORMALIZATION_ENABLED`
  independently to isolate retrieval impact from extraction impact.

**Decision criterion to drop the legacy path:** new path matches or beats
legacy on `missile_propulsion` ✓ exact at T=1.0 across two independent
re-ingest runs, AND doesn't regress any other pass's ✓ exact count by more
than 1.

**Container rebuild posture** (per `feedback_container_rebuild`):
- `app/services/table_normalization/` is under bind-mounted `./app` — no
  rebuild needed for `api`/`worker`/`worker-graph`.
- `docker compose build docling-graph` required to pick up the new module
  (COPY into image).

## 14. Test Posture

| Layer | Coverage | Location |
|---|---|---|
| Detection unit tests | SA-2 hybrid; plain column-major; row-major; undersized 3×3; malformed flags; section headers | `tests/unit/test_table_normalization_detect.py` |
| Normalization unit tests | Empty cells skipped; merged cells expanded; units extracted; multi-value cells preserved; idempotency | `tests/unit/test_table_normalization_normalize.py` |
| Graph renderer unit + snapshot | SA-2 column chunks; small whole; long-column section-split; OTHER fallback | `tests/unit/test_table_normalization_render_graph.py` + JSON fixture |
| Embedding renderer unit + snapshot | Always-summary; size-aware whole/column; format identical to graph | `tests/unit/test_table_normalization_render_embedding.py` + JSON fixture |
| Token sizing | bge-m3 tokenizer agrees with chunk-size budget within ±5% | `tests/unit/test_table_normalization_tokens.py` |
| Storage round-trip | Write/read `chunk_metadata`; partial index | `tests/integration/test_chunk_metadata_persistence.py` |
| Retrieval surfacing | `/v1/retrieval/query` returns `table_chunk` block | `tests/integration/test_retrieval_table_chunk_surfacing.py` |
| Graph provenance | `ExtractionFieldProvenance.evidence_ids` contains underlying cell refs | `tests/integration/test_graph_provenance_cell_refs.py` |
| End-to-end SA-2 | Ingest → extract → score → cell_refs traceable | `tests/integration/test_sa2_table_pipeline_e2e.py` |
| Legacy fallback | Flag flip preserves today's `_table_facts.py` behavior | `tests/integration/test_legacy_table_facts_fallback.py` |

**Verification additions to `VERIFICATION_CHECKLIST.md`:**
- `□ Normalize SA-2 doc → ≥ 3 NormalizedTable.shape == COLUMN_MAJOR|HYBRID`
- `□ Embedding ingest SA-2 → query "S-75M2 max range" → top-1 result `chunk_kind == "table_entity_column"``
- `□ Graph extract SA-2 (new path) → score against GT → record ✓ exact for missile_propulsion`
- `□ DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS=true → re-ingest+extract → score matches prior baseline (regression-free dormant path)`
- `□ `.env` + `.env.example` contain all 8 new variables`

## 15. Success Criteria (acceptance gates)

1. **Functional:**
   - SA-2 sample table → `Shape.HYBRID`, 4 identity rows, 2+ sections, all
     data columns extracted to `NormalizedColumn`.
   - Graph renderer → one chunk per entity column on SA-2; chunk text
     matches committed snapshot.
   - Embedding renderer → 1 `table_summary` + N `table_entity_column`
     chunks for SA-2 (whole-table NOT emitted; SA-2 > 512 tokens).
2. **Provenance:**
   - `/v1/retrieval/query` for a variant-specific query → top-1 result with
     populated `table_chunk` block and correct `cell_refs`.
   - Extracted `Missile.max_range_m` on `S-75M2` →
     `ExtractionFieldProvenance.evidence_ids` contains both graph-chunk
     self_ref AND underlying cell ref.
3. **Regression-free legacy path:**
   - With `DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS=true`, SA-2 extraction
     output matches a fixture captured against `main` before this rewrite,
     ±0 ✓ exact differences.
4. **Master kill-switch correctness:**
   - With both `*_NORMALIZATION_ENABLED=false`, chunk emissions on both
     sides are identical to a pre-rewrite fixture.
5. **Quality (post-merge measurement, not merge-gating):**
   - SA-2 `missile_propulsion` ✓ exact ≥ 6 at T=1.0 with the new path. If
     lower, flip legacy flag and file a follow-up; do not block merge.

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **SA-2 propulsion regression.** Per-column chunks push row-label→schema-field mapping back onto the LLM — the failure mode `_table_facts.py` was built to fix. New path may regress `missile_propulsion` ✓ exact below the validated baseline. | Dormant `_table_facts.py` + `_alias_map.py` behind `DOCLING_GRAPH_USE_LEGACY_TABLE_FACTS`. One flag flip restores baseline without redeploy. Decision criterion in §13. |
| **Token-budget defaults wrong for some document classes.** SA-2 was the basis for the analysis; other table-heavy docs may have different shapes. | All budgets are env-tunable. Diagnostics capture `tables_fit_whole` / `tables_split_by_column` / `by_kind` per document — query to spot anomalies. |
| **Shape detection misclassifies a real table as OTHER.** Heuristic floor (4×4) and reliance on `row_header` / `column_header` flags assume clean Docling parsing. | OTHER fallback is *additive-not-destructive*: raw markdown preserved as a single chunk. No table disappears; worst case is "no improvement vs today." Snapshot tests guard against drift. |
| **Hybrid table identity rows over-merge.** Two genuinely different concepts both looking "identity-shaped" could collapse into one identity dict. | Detection rule requires identity rows at the *top* of the table; section headers and spec rows interleaved with identity rows prevent over-merging. Snapshot test for constructed worst-case fixture. |
| **`chunk_metadata` unused by older consumers.** Older retrieval consumers may not surface `cell_refs`. | Additive `table_chunk` block in response envelope; consumers that don't know about it ignore it. |
| **ArcadeDB `chunk_kind` index migration.** Adding indexed property to existing vertex schema may require rebuild. | Blow-away + re-ingest covers this — no in-place schema migration needed. |
| **`bge-m3` tokenizer load adds startup cost.** First call loads HF tokenizer (~tens of MB). | Lazy-load on first use; cached at module level for process lifetime. Same model the embedding pipeline already pulls. |
| **Empty `display_name` for columns missing all identity fields.** Some tables have malformed key columns. | Stable fallback `"col-{col_idx}"`. Chunk still emits with traceable `column_index` + `cell_refs`. |
| **Lifetime drift of dormant `_table_facts.py`.** Code that doesn't run rots; the fallback could become fictional. | Integration test `test_legacy_table_facts_fallback.py` runs the legacy path on every CI build. Failure surfaces drift immediately. |

## 17. Known Limitations

- Row-major tables supported but not primary focus. One snapshot test;
  production performance to be measured.
- `display_name` heuristic is hand-coded. Other table styles may need
  additions (one-line change + test).
- Section keyword list is hand-coded. Detection diagnostics surface
  misses (`section_headers_detected: 0` on a table with sections).
- `cell_refs` JSON-Pointer format assumes Docling's `data.table_cells`
  array structure is stable. If Docling changes that schema, pointers
  stale. Mitigated by capturing `table_index` + `(row_idx, col_idx)`
  redundantly inside the JSON-Pointer string.

## 18. Open Questions

None at design time. All decisions locked through brainstorming clarifying
questions (2026-05-11 conversation). Implementation plan to be created
by the writing-plans skill after this design is approved.
