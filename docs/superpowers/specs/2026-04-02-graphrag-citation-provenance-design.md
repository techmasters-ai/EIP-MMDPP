# GraphRAG Citation Provenance — Design Spec

**Goal:** Add inline citations to all four GraphRAG search responses (Local, Global, Drift, Basic) that trace facts back through entities/relationships to source document text, with clickable citation links in the UI.

**Status:** Design approved, pending implementation plan.

---

## Provenance Chain

```
Document (Postgres) → text_unit (GraphRAG chunk) → entity/relationship → community_report → LLM response
```

All links are ID-based through existing Parquet data. No fuzzy matching or page-level resolution needed.

---

## 1. Prompt Changes

### Files modified
- `app/services/graphrag_prompts.py` — all 5 prompt functions:
  - `get_global_search_map_prompt()`
  - `get_global_search_reduce_prompt()`
  - `get_local_search_prompt()`
  - `get_drift_search_prompt()`
  - `get_basic_search_prompt()`

### Instruction added to each prompt

Each prompt gets an additional instruction block telling the LLM to:
1. Insert inline `[n]` citations (sequential numbering) referencing the specific entity or relationship that supports each claim
2. Append a machine-parseable `## Sources` block at the end listing each citation number with the entity/relationship IDs it references

### Expected LLM output format

```
The SA-2 Guideline uses command guidance [1] with the Fan Song radar
providing target tracking [2].

## Sources
[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276
[2] Entity: SNR-75 FAN SONG (1494), Relationship: 3494
```

The `## Sources` block is stripped from the user-facing response after extraction.

---

## 2. Citation Resolver (post-processing)

### New function: `_resolve_citations(response_text, data) -> (clean_text, sources)`

Located in `app/services/graphrag_service.py`.

**Steps:**

1. **Parse** — regex extracts the `## Sources` block. Each line maps a citation number to entity/relationship `human_readable_id` values.

2. **Strip** — removes the `## Sources` block from response text. User sees clean prose with `[n]` markers only.

3. **Resolve** — for each cited ID, look up in the already-loaded Parquet DataFrames:
   - Entity: `title`, `type`, `description` (from `entities.parquet`)
   - Relationship: `source`, `target`, `description` (from `relationships.parquet`)
   - From both: `text_unit_ids` → look up `text_units.parquet` → get `text` and `document_id`
   - From `document_id`: look up document title from the GraphRAG input CSV

4. **Build sources array** — one entry per citation number:

```json
{
  "citation": 1,
  "entities": [
    {"id": 3349, "title": "SA-2 GUIDELINE", "type": "MISSILE_SYSTEM", "description": "..."}
  ],
  "relationships": [
    {"id": 4276, "source": "S-75 DVINA", "target": "V-750", "description": "..."}
  ],
  "source_documents": [
    {
      "document_id": "34d9606f-...",
      "document_title": "Red SAM: The SA-2 Guideline",
      "source_text": "The SA-2 Guideline (S-75 Dvina) surface-to-air..."
    }
  ]
}
```

### Graceful degradation

If the LLM fails to produce a `## Sources` block (malformed output, thinking model edge case), `sources` returns as an empty array and `content_text` is returned as-is. No hard failure.

### Data access

All Parquet data is already loaded by `_load_search_data()`. The resolver receives the loaded `data` dict — no additional I/O.

---

## 3. API Response Shape

### Modified files
- `app/services/graphrag_service.py` — `local_search()`, `global_search()`, `drift_search()`, `basic_search()` call `_resolve_citations()` before returning
- `app/workers/graphrag_tasks.py` — serializes the new `sources` field

### Response payload

```json
{
  "score": 1.0,
  "modality": "graphrag_response",
  "content_text": "The SA-2 Guideline uses command guidance [1]...",
  "context": {
    "source": "graphrag_global",
    "sources": [
      {
        "citation": 1,
        "entities": [
          {"id": 3349, "title": "SA-2 GUIDELINE", "type": "MISSILE_SYSTEM", "description": "..."}
        ],
        "relationships": [
          {"id": 4276, "source": "S-75 DVINA", "target": "V-750", "description": "..."}
        ],
        "source_documents": [
          {
            "document_id": "34d9606f-...",
            "document_title": "Red SAM: The SA-2 Guideline",
            "source_text": "The SA-2 Guideline (S-75 Dvina) surface-to-air..."
          }
        ]
      }
    ],
    "graphrag_context": { "..." : "raw context preserved for backwards compatibility" }
  }
}
```

### Backwards compatibility

`graphrag_context` is preserved as-is. `sources` is a new additive field. Existing consumers are unaffected.

---

## 4. Frontend — Clickable Citations

### Modified files
- The component that renders `graphrag_response` modality results in the query results page

### Behavior
- Before rendering, replace `[n]` markers in `content_text` with clickable anchor links
- Below the response text, render a **Sources** panel listing each citation with entity names, relationship descriptions, and source document titles
- Clicking a citation number scrolls to / highlights the corresponding source entry

### No new pages or routes needed.

---

## 5. Microsoft GraphRAG Coupling

### What we change (ours only):
- Prompt templates in `app/services/graphrag_prompts.py`
- Post-processing in `app/services/graphrag_service.py`
- Response shaping in `app/workers/graphrag_tasks.py`
- Frontend rendering

### What we do NOT change:
- Any code inside `graphrag`, `graphrag_llm`, `graphrag_cache`, `graphrag_input`, `graphrag_storage` packages
- GraphRAG API function signatures
- Parquet schema or indexing pipeline

### Coupling risk
We depend on the `(response, context)` tuple from search functions and Parquet column names (`human_readable_id`, `text_unit_ids`, `document_id`). A major GraphRAG version bump could require resolver updates — same risk the existing code already carries.

Custom prompts are already fully overridden (military ontology), so GraphRAG updates won't overwrite them.

---

## 6. Testing Strategy

### Unit tests
- `_resolve_citations()` — parse various `## Sources` block formats, handle malformed output, handle empty citations, verify Parquet lookups return correct entities/relationships/documents
- `_strip_sources_block()` — verify clean extraction and removal from response text
- Prompt content tests — verify each search prompt contains citation instructions

### Integration tests
- Mock LLM returning response with `[n]` markers and `## Sources` block
- Verify full pipeline: search function → citation resolution → API response with populated `sources` array
- Verify graceful degradation when LLM produces no citations

### Manual verification
- Run all 4 search types against live graph
- Confirm citations resolve to real entities
- Verify frontend renders clickable citations
