# Document Structure Extraction Pass — Design

**Date:** 2026-04-21
**Branch target:** `feat/pydantic-ontology-ssot` (or follow-on feature branch)
**Bundle:** `air_defense_v3`
**Task IDs:** #1 (architecture), #2 (schemas), #3 (walker), #4 (TextBlockEntity + NEAR_TEXT), #5 (tests)

## Problem

The current `air_defense_v3` bundle runs three LLM extraction passes (`radar_domain`, `missile_domain`, `system_links`) over the document markdown. Nothing in the pipeline explicitly graphs the document's **own structural entities** — DOCUMENT, SECTION, FIGURE, TABLE, IMAGE — even though Docling already parsed all of them into `DoclingDocument.texts[]`, `doc.pictures[]`, and `doc.tables[]` with page numbers, self_refs, captions, and parent-ref chains.

Consequences today:
- Downstream retrieval has no graph-level way to answer "what figures does section 3 reference?" or "show me every image in this document with its surrounding context."
- The Phase 8 provenance work wires `element_uid` on extracted entities but there is no corresponding graph node for the element itself — provenance points at parser locations that never become vertices.
- An LLM pass attempting to extract this structural information from markdown would hallucinate section numbers and miss embedded pictures; the parser already has the right answer.

## Goal

Add a new extraction pass to `air_defense_v3` that emits DOCUMENT, SECTION, FIGURE, TABLE, IMAGE, and TEXT_BLOCK entities (plus their typed edges) **deterministically from the DoclingDocument + API `Document` SQL row**. No LLM call.

Out of scope for this change:
- Re-wiring downstream passes (`radar_domain`, `missile_domain`, `system_links`) to depend on the new pass. Refs will be available but unused until a follow-on change.
- Emitting `MENTIONED_IN` / `APPEARS_IN` edges from domain entities (RADAR_SYSTEM, MISSILE_SYSTEM) back to structural entities. Follow-on.
- Any new graph viz work. The existing viewer picks up new vertex/edge types automatically.

## Architecture (Section 1)

**New endpoint `/extract-structure` on the docling-graph service.** The endpoint takes the same request shape as `/extract-pass` (bundle_key, pass_name, document_id, docling_document_json, plus a new optional `document_metadata` dict for the SQL row fields) and returns the same `ExtractPassResponse` shape, but runs a Python walker instead of `run_pipeline`. No LLM call, no quality gate, no batch errors.

**Manifest gains a new field `extraction_method`** on every pass. Default is `"llm"` (existing passes are unchanged). `document_structure` sets `extraction_method: "structural"`. The worker's `_call_extract_pass` routes by this flag:

- `extraction_method: "llm"` (or missing) → POST to `/extract-pass` (existing behavior)
- `extraction_method: "structural"` → POST to `/extract-structure`

Both endpoints return the same response schema. Worker's downstream code (`_parse_pass_response`, `_extend_upstream_refs`, walker summary) is unchanged.

Manifest ordering: `document_structure` becomes the first pass. `radar_domain` / `missile_domain` / `system_links` shift down. `depends_on` stays as-is on all three.

**Why this over a worker-side pre-pass step?** Preserves the "every pass is declared in manifest.yaml" invariant. Downstream passes get a real `depends_on` target (even if nothing uses it in this change). The LLM-vs-code distinction is an implementation detail kept inside the service.

## Entities and schemas (Section 2)

Four entities (DocumentEntity, SectionEntity, FigureEntity, TableEntity) already exist in `ontology_bundles/air_defense_v3/entities.py`. **Additions:**

### `ImageEntity` (new)

```python
class ImageEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="IMAGE",
        graph_id_fields=["image_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )
    image_ref: str  # identity — self_ref like "#/pictures/12"
    document_id: Optional[str]
    page: Optional[int]
    caption: Optional[str]
    mime_type: Optional[str]
    storage_key: Optional[str]  # MinIO key
    bbox: Optional[dict]        # {l, t, r, b, page, coord_origin}
    image_role: Optional[str]   # HEADER_LOGO | INLINE_IMAGE | UNCAPTIONED_FIGURE
    confidence: Optional[float] = 1.0
    near_text: List["TextBlockEntity"] = edge("NEAR_TEXT", default_factory=list)
```

### `TextBlockEntity` (new)

```python
class TextBlockEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="TEXT_BLOCK",
        graph_id_fields=["text_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )
    text_ref: str  # identity — self_ref like "#/texts/237"
    document_id: Optional[str]
    text: Optional[str]          # truncated at 500 chars
    label: Optional[str]         # TEXT | PARAGRAPH | LIST_ITEM
    page: Optional[int]
    confidence: Optional[float] = 1.0
```

**Lazy emission:** TEXT_BLOCK is never emitted as a top-level list on the pass root. It only appears as a child under IMAGE or FIGURE's `near_text` field. A text block that nothing references never enters the graph.

### Additions to existing entities

- `DocumentEntity` gains `storage_key: Optional[str]` (source file MinIO key).
- `FigureEntity` gains `storage_key: Optional[str]` (figure image MinIO key) and `near_text: List[TextBlockEntity] = edge("NEAR_TEXT", default_factory=list)`.

### `DocumentStructurePass` (new pass template)

File: `ontology_bundles/air_defense_v3/extraction_schemas/document_structure.py`

```python
class DocumentStructurePass(BaseModel):
    model_config = ConfigDict(is_entity=True, graph_id_fields=[])

    document: Optional[DocumentEntity] = edge(label="DESCRIBES")
    sections: List[SectionEntity]      = edge(label="CONTAINS_SECTION", default_factory=list)
    figures:  List[FigureEntity]       = edge(label="CONTAINS_FIGURE",  default_factory=list)
    tables:   List[TableEntity]        = edge(label="CONTAINS_TABLE",   default_factory=list)
    images:   List[ImageEntity]        = edge(label="CONTAINS_IMAGE",   default_factory=list)
```

### Pass classification

```yaml
- name: document_structure
  required: true
  kind: entities_and_relationships
  input_mode: document_only
  extraction_method: structural   # NEW field
  module: extraction_schemas.document_structure
  template_class: DocumentStructurePass
  primary_entity_types: [DOCUMENT, SECTION, FIGURE, TABLE, IMAGE]
  bridge_entity_types: []
  extracted_relationship_types: [CONTAINS_SECTION, CONTAINS_FIGURE, CONTAINS_TABLE, CONTAINS_IMAGE, NEAR_TEXT, DESCRIBES]
  depends_on: []
```

TEXT_BLOCK is intentionally not listed in `primary_entity_types` because it's emitted lazily as an edge child, not a top-level entity.

## Data flow & relationships (Section 3)

```
upload → Docling convert → DoclingDocument + Document SQL row → ingest pipeline
       ↓
       pass 1: document_structure  (NEW)
         worker POST /extract-structure with doc_json + document metadata
         walker emits 5 entity lists + typed edges + TextBlockEntity children
         worker runs _extend_upstream_refs → upstream_refs has E001..E00N
         covering every DOCUMENT + SECTION + FIGURE + TABLE + IMAGE
       ↓
       pass 2: radar_domain   (existing, unchanged)
       pass 3: missile_domain (existing, unchanged)
       pass 4: system_links   (existing, unchanged)
```

### Walker logic

1. **DOCUMENT** — one entity. `document_number` from SQL row `title` (with filename fallback), `document_id` from SQL row id, `storage_key` from SQL row, `source_type` inferred from mime_type (PDF → MANUAL, XLSX → SPREADSHEET, etc.), `publication_date` from SQL row or doc origin.

2. **SECTION** — one per `doc.texts[i]` with `label == DocItemLabel.SECTION_HEADER`. `section_number` extracted via regex `^(\d+(?:\.\d+)*)\s`; falls back to `self_ref` on miss. `section_path` built from the text's parent chain. `page_start` from first prov page; `page_end` computed from the next section's `page_start - 1` (or doc last page for the final section).

3. **Picture fork (FIGURE vs IMAGE)** — for each `doc.pictures[i]`:
   - Caption looked up via `pic.captions[0].cref` → `doc.texts[...]`.
   - If caption matches `^(Figure|Fig\.?)\s+[\d\w\-\.]+` → `FigureEntity(figure_ref=<captured>, ...)`.
   - Else → `ImageEntity(image_ref=pic.self_ref, image_role=<heuristic>)`:
     - First picture on page 1 with bbox top-region + area < 10% page area → `HEADER_LOGO`.
     - Has caption but doesn't match "Figure N" → `UNCAPTIONED_FIGURE`.
     - Otherwise → `INLINE_IMAGE`.

4. **TABLE** — one per `doc.tables[i]` with same caption-regex logic for `table_ref`.

5. **Surrounding text** — for each emitted IMAGE or FIGURE, walk reading order from the picture's position ±2 text blocks, skipping: section headers, captions already attached to pictures/tables, other pictures, other tables. Emit up to 4 `TextBlockEntity` instances as children of that IMAGE's/FIGURE's `near_text` field. Dedup across pictures by `text_ref` — the same text block shared by two pictures emits one `TextBlockEntity` (per-picture edges remain).

### Typed edges emitted

| From | Relationship | To | Source |
|---|---|---|---|
| PASS_ROOT | DESCRIBES | DOCUMENT | pass-root field |
| PASS_ROOT | CONTAINS_SECTION | SECTION | pass-root field |
| PASS_ROOT | CONTAINS_FIGURE | FIGURE | pass-root field |
| PASS_ROOT | CONTAINS_TABLE | TABLE | pass-root field |
| PASS_ROOT | CONTAINS_IMAGE | IMAGE | pass-root field |
| SECTION | CONTAINS_SECTION | SECTION | parent-chain walk (sub-section nesting) |
| SECTION | CONTAINS_FIGURE | FIGURE | `pic.parent` chain → nearest SECTION_HEADER; page-range fallback |
| SECTION | CONTAINS_TABLE | TABLE | same |
| SECTION | CONTAINS_IMAGE | IMAGE | same |
| IMAGE | NEAR_TEXT | TEXT_BLOCK | reading-order ±2 |
| FIGURE | NEAR_TEXT | TEXT_BLOCK | reading-order ±2 |

## Error handling & diagnostics (Section 4)

Deterministic walker; the error surface is limited to parser-output edge cases.

| Case | Walker behavior |
|---|---|
| Empty DoclingDocument | Return pass with DOCUMENT only; lists empty. 200 OK. |
| section_number regex miss | Fall back to `self_ref`; increment `section_number_regex_misses`. |
| Picture has no captions | Classify as IMAGE; `caption=None`; role per heuristic. |
| Caption ref points to non-existent text item | Log WARNING, treat as uncaptioned; increment `caption_lookup_misses`. |
| Picture has no self_ref | Log WARNING, skip picture; append to `walker_warnings`. |
| `pic.parent` chain doesn't reach SECTION_HEADER | Page-range fallback (`page_start ≤ pic.page`); increment `parent_chain_fallbacks`. |
| Document SQL row missing storage_key/title | Walker populates what it has; fall back identity to `document_id`. |
| Pydantic validation failure | PassTerminal — log dict + error; return 500. Indicates walker bug, not runtime condition. |
| Unhandled walker exception | Caught at endpoint; returned as 200 with `diagnostics.pipeline_error = {type, message}` — same pattern `/extract-pass` just gained. |

### Diagnostics payload

```python
diagnostics = {
    "extraction_method": "structural",
    "walker_stats": {
        "texts_scanned": 847,
        "sections_emitted": 12,
        "pictures_scanned": 28,
        "figures_emitted": 9,
        "images_emitted": 19,
        "tables_scanned": 6,
        "tables_emitted": 6,
        "text_blocks_emitted": 42,   # unique TextBlockEntities after dedup
        "section_number_regex_misses": 2,
        "caption_lookup_misses": 0,
        "parent_chain_fallbacks": 3,
    },
    "walker_warnings": [
        "picture #/pictures/7 caption ref #/texts/412 not found — treated as uncaptioned",
    ],
    "library_log": "",         # empty; structural pass never calls the library
    "pipeline_error": None,    # populated only on unhandled exception
}
```

The existing notebook walker cell (`ingest_walkthrough.ipynb` id `6bfbd202`) already renders `diagnostics` generically — it will pick up `walker_stats` and `walker_warnings` automatically. One-line header added to distinguish `extraction_method=structural` from `extraction_method=delta` so readers know why there's no LLM fallback chain to inspect.

## Testing (Section 5)

### Unit tests — `tests/ontology_bundles/air_defense_v3/test_document_structure_walker.py`

- `test_empty_docling_document` — walker returns DOCUMENT only, no crash.
- `test_document_entity_fields` — SQL row fields plumbed correctly.
- `test_section_hierarchy` — 3-level nested sections emit correct hierarchy + SECTION→SECTION edges.
- `test_section_number_regex_fallback` — non-numeric heading → self_ref identity.
- `test_figure_vs_image_classification` — three pictures, correct sort.
- `test_header_logo_heuristic` — page 1 + small top-region bbox → HEADER_LOGO.
- `test_table_extraction` — two tables with captions, regex extracts `table_ref`.
- `test_near_text_window` — exactly 4 TextBlockEntities per image, skipping non-text items.
- `test_near_text_dedup` — shared paragraph emits one TextBlockEntity, attached to two images.
- `test_section_attribution_parent_chain` — clean chain → no page-range fallback.
- `test_section_attribution_page_fallback` — broken chain → fallback counted in diagnostics.
- `test_pydantic_validation_failure_surfaces` — corrupt picture skipped, warning in `walker_warnings`, 200 response.

### Integration — `tests/integration/test_document_structure_pipeline.py`

- POST `/extract-structure` with `S-75 Dvina.pdf` DoclingDocument JSON.
- Assert ≥1 DOCUMENT, ≥3 SECTIONs, ≥1 FIGURE, ≥1 IMAGE, ≥1 TABLE; each IMAGE/FIGURE has `near_text`.
- Assert response time < 5 seconds.

### End-to-end — `tests/integration/test_full_ingest_with_structure_pass.py`

- Full worker pipeline on one real PDF.
- After `document_structure` pass, `upstream_refs` has E001..E00N covering all 6 entity types.
- Merged ArcadeDB graph has the expected vertices + CONTAINS_* edges.

### Fixtures

- Hand-built minimal DoclingDocument dict (fast unit tests).
- `S-75 Dvina.pdf` → JSON, committed to `tests/fixtures/`.
- `Fan_Song_Radar.jpeg` → JSON via synthetic-picture fallback (degenerate edge case).

### Post-implementation

Per project convention (memory `feedback_post_code_workflow.md`):
1. Run simplify on all new code.
2. Full test suite.
3. Update `VERIFICATION_CHECKLIST.md` with the new pass.
4. Update README if the public pass list changes.

## File-level change summary

| File | Change |
|---|---|
| `ontology_bundles/air_defense_v3/entities.py` | Add `ImageEntity`, `TextBlockEntity`; add `storage_key` to `DocumentEntity` + `FigureEntity`; add `near_text` to `FigureEntity` + `ImageEntity`. |
| `ontology_bundles/air_defense_v3/extraction_schemas/document_structure.py` | New — `DocumentStructurePass` template. |
| `ontology_bundles/air_defense_v3/extraction_schemas/__init__.py` | Register new module. |
| `ontology_bundles/air_defense_v3/manifest.yaml` | Add `document_structure` pass as pass #1; add `extraction_method: llm` to existing three passes for explicit routing. |
| `ontology_bundles/air_defense_v3/introspect.py` (if it enumerates passes) | Pick up the new pass in ontology dict derivation. |
| `docker/docling-graph/app/main.py` | New `/extract-structure` endpoint; shared response-building helpers with `/extract-pass`. |
| `docker/docling-graph/app/structural_walker.py` | New — Python walker implementation. |
| `docker/docling-graph/app/schemas.py` | Add `DocumentMetadata` request sub-model for SQL row fields. |
| `app/workers/pipeline.py` | `_call_extract_pass` router by `pass_def.extraction_method`. |
| `app/bundles/manifest.py` | Parse the new `extraction_method` field (default `"llm"`). |
| `notebooks/ingest_walkthrough.ipynb` | Minor: header line distinguishing structural vs LLM passes in the per-pass diagnostics dump. |
| `tests/...` | Unit + integration + E2E tests per Section 5. |
| `VERIFICATION_CHECKLIST.md` | New pass entry. |

## Decisions deferred

- Downstream consumers of the new structural refs (e.g. `radar_domain` emitting `RADAR_SYSTEM --MENTIONED_IN--> SECTION`). Follow-on change.
- `image_role` enum expansion (currently 3 values). If real docs surface a fourth obvious role, extend.
- Window size for `near_text` (currently 2 before + 2 after). If RAG evaluation shows a different N works better, tune in a follow-on.
