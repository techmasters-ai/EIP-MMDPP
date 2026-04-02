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

## 1. Prompt Changes — Per-Strategy Citation Format

The LLM receives different context data depending on the search type. The citation format must match what the LLM can actually see.

### Files modified
- `app/services/graphrag_prompts.py` — all 5 prompt functions

### Strategy-specific citation formats

| Search Type | LLM Sees Entity IDs? | Citation Format | Resolution Method |
|---|---|---|---|
| **Local** | Yes (`short_id` in context tables) | `[n] Entity: NAME (ID), Relationship: ID` | Look up by `human_readable_id` (exposed as `short_id` in GraphRAG data model) |
| **Drift** | Yes (same as Local) | `[n] Entity: NAME (ID), Relationship: ID` | Look up by `human_readable_id` (exposed as `short_id` in GraphRAG data model) |
| **Global** | No (only community report text) | `[n] Entity: NAME` | Match by `title` in entities DataFrame |
| **Basic** | No (only raw text chunks) | `[n] Source: "first 50 chars of text..."` | Match by text content in text_units DataFrame |

### Instruction placement

Citation instructions are added to each prompt except `get_global_search_map_prompt()`. The map prompt extracts key points from individual community reports — citations belong in the **reduce** prompt where the LLM synthesizes the final answer. Adding citations to the map prompt would produce nested/conflicting citation formats.

### Instruction added to each prompt (except global map)

Each prompt gets an additional instruction block telling the LLM to:
1. Insert inline `[n]` citations (sequential numbering) referencing the specific source that supports each claim
2. Append a machine-parseable `## Sources` block at the end listing each citation

### Expected LLM output — Local/Drift (numeric IDs)

```
The SA-2 Guideline uses command guidance [1] with the Fan Song radar
providing target tracking [2].

## Sources
[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276
[2] Entity: SNR-75 FAN SONG (1494), Relationship: 3494
```

### Expected LLM output — Global (name-based)

```
The SA-2 Guideline uses command guidance [1] with the Fan Song radar
providing target tracking [2].

## Sources
[1] Entity: SA-2 GUIDELINE
[2] Entity: SNR-75 FAN SONG
```

### Expected LLM output — Basic (text-based)

```
The SA-2 system uses command guidance for missile control [1].

## Sources
[1] Source: "The SA-2 Guideline (S-75 Dvina) surface-to-air missile system uses..."
```

The `## Sources` block is stripped from the user-facing response after extraction.

---

## 2. Citation Resolver (post-processing)

### New function: `_resolve_citations(response_text, data, strategy) -> (clean_text, sources)`

Located in `app/services/graphrag_service.py`. Takes `strategy` to determine which resolution method to use.

**Steps:**

1. **Parse** — regex extracts the `## Sources` block. Each line maps a citation number to its references (IDs, names, or text depending on strategy).

2. **Strip** — removes the `## Sources` block from response text. Also strips any `<think>`/`<thinking>` tags that may bleed through from thinking models. User sees clean prose with `[n]` markers only.

3. **Resolve** — strategy-dependent lookup:
   - **Local/Drift:** look up entity/relationship by `human_readable_id` in Parquet DataFrames
   - **Global:** match entity by `title` (case-insensitive) in entities DataFrame
   - **Basic:** match text_unit by substring against `text` column in text_units DataFrame
   - For all: follow `text_unit_ids` → `text_units` DataFrame → get `text` and `document_id` → `documents` DataFrame → get document title

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

For **Basic** search, `entities` and `relationships` arrays will be empty; only `source_documents` is populated.

### Edge case handling

| Scenario | Behavior |
|---|---|
| No `## Sources` block produced | Return empty `sources` array, `content_text` as-is |
| Citation references non-existent ID | Skip that citation entry, log warning |
| Malformed citation line | Skip that line, log debug |
| Duplicate citation numbers | Keep first occurrence |
| `<think>` tags in response | Strip before parsing |
| Global search name match fails | Skip that citation, log warning |
| Basic search text match fails | Skip that citation, log warning |
| text_unit has null `document_id` | Omit `source_documents` for that citation (return empty array) |

### Data access

`_load_search_data()` must be updated to also load `documents.parquet`:

```python
for name in (
    "entities", "communities", "community_reports",
    "text_units", "relationships", "covariates",
    "documents",  # NEW — needed for document title resolution
):
```

Document titles from the bridge layer include a content hash suffix (e.g., `"Red SAM_a3b2c1d4"`). The resolver strips the `_<hex>` suffix before returning `document_title`.

---

## 3. API Response Shape

### Modified files
- `app/services/graphrag_service.py` — all four search functions call `_resolve_citations()` before returning
- `app/workers/graphrag_tasks.py` — serializes the new `sources` field

### Response payload (uniform across all 4 search types)

```json
{
  "results": [
    {
      "score": 1.0,
      "modality": "graphrag_response",
      "content_text": "The SA-2 Guideline uses command guidance [1]...",
      "classification": "UNCLASSIFIED",
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
  ],
  "query_text": "How does the SA-2 guidance system work?",
  "strategy": "graphrag_global"
}
```

### Backwards compatibility

`graphrag_context` is preserved as-is. `sources` is a new additive field. Existing consumers are unaffected.

### Streaming note

Citation resolution requires the full response text. It is incompatible with token-by-token streaming. All four search types currently collect the full response before returning, so this is not a constraint today. If streaming is added in the future, citation resolution must run as a post-stream step.

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
- `_resolve_citations()` — all three resolution strategies (ID-based, name-based, text-based), handle malformed output, missing IDs, empty citations, duplicate citations
- `_strip_sources_block()` — verify clean extraction and removal, including `<think>` tag stripping
- Prompt content tests — verify each of the 5 search prompts contains citation instructions appropriate to its strategy

### Integration tests
- Mock LLM returning response with `[n]` markers and `## Sources` block for each strategy
- Verify full pipeline: search function → citation resolution → API response with populated `sources` array
- Verify graceful degradation when LLM produces no citations (empty `sources`, no crash)
- Verify Basic search returns `source_documents` with empty `entities`/`relationships`

### Manual verification
- Run all 4 search types against live graph
- Confirm citations resolve to real entities/documents
- Verify frontend renders clickable citations
