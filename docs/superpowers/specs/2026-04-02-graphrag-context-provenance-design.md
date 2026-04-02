# GraphRAG Context-Based Provenance Design Spec

## Problem

The LLM-based citation system is fragile. It relies on the LLM producing citations in a specific format (`## Sources` block or `[Data: Sources (...)]` markers), but the GraphRAG library's own prompts override our instructions, producing inconsistent formats that the parser frequently fails to resolve. When parsing fails, citations are either empty or point to wrong entities.

## Goal

Replace LLM-based citations with deterministic, context-based provenance. Every GraphRAG response already has the full context that was fed to the LLM (entities, relationships, community reports, text units). Surface this context in a structured `provenance` key in the API response, organized hierarchically by community report, so users and downstream applications can cross-check any claim against the original source material.

## Design

### API Response Structure

The `provenance` key is an array of community reports. Each report contains its member entities, relationships, text_units, and covariates. Each leaf item includes the source documents it was extracted from.

`graphrag_context` is preserved for existing consumers. The `sources` key (LLM-parsed citations) is removed.

```json
{
  "score": 1.0,
  "modality": "graphrag_response",
  "content_text": "Fan Song is the engagement radar...",
  "classification": "UNCLASSIFIED",
  "context": {
    "source": "graphrag_local",
    "graphrag_context": { "...existing serialized context..." },
    "provenance": [
      {
        "report_id": "11",
        "report_title": "SA-2 Guideline & Fan Song Radar Community",
        "report_content": "# SA-2 Guideline & Fan Song Radar...\n\nThe community centers on...",
        "entities": [
          {
            "id": 3349,
            "title": "FAN SONG",
            "type": "FIRE_CONTROL_SYSTEM",
            "description": "Engagement radar for S-75/SA-2...",
            "source_documents": [
              {"document_id": "doc-1", "document_title": "Red SAM"}
            ]
          }
        ],
        "relationships": [
          {
            "id": 4276,
            "source": "S-75 DVINA",
            "target": "V-750",
            "description": "Interceptor component",
            "source_documents": [
              {"document_id": "doc-1", "document_title": "Red SAM"}
            ]
          }
        ],
        "text_units": [
          {
            "id": 0,
            "text": "The SA-2 Guideline uses command guidance...",
            "source_documents": [
              {"document_id": "doc-1", "document_title": "Red SAM"}
            ]
          }
        ],
        "covariates": []
      }
    ]
  }
}
```

### Per-Strategy Behavior

| Strategy | provenance content |
|----------|-------------------|
| **Local** | Community reports with entities, relationships, text_units, covariates grouped under each report |
| **Drift** | Same as Local (Drift uses LocalSearch internally) |
| **Global** | Community reports with report_content only. The global search context builder does **not** include entities, relationships, or text_units — only report DataFrames. Entity/relationship/text_unit lists will be empty for every report. |
| **Basic** | Single entry with no report info; text_units populated from the context `sources` key with source documents resolved |

### Backend Module: `app/services/graphrag_provenance.py`

New module with a single public function:

```python
def build_provenance(
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict]:
```

**Parameters:**
- `context` — the context_records dict returned by GraphRAG's search (second element of the tuple). Known keys vary by strategy:
  - Local/Drift: `"reports"`, `"entities"`, `"relationships"`, `"sources"` (text units), `"claims"` (covariates)
  - Global: `"reports"` only
  - Basic: `"sources"` only (text units)
  - Note: covariates are keyed as `"claims"` (not `"covariates"`) in the context dict.
- `data` — the full loaded Parquet data from `_load_search_data()`, including `communities`, `community_reports`, `text_units`, and `documents` DataFrames needed for resolution.
- `strategy` — the search strategy string, used to handle per-strategy differences.

**ID type mismatch:** The context DataFrames use `human_readable_id` (integer) as their `"id"` column, but `data["communities"]` stores `entity_ids` and `relationship_ids` as UUID strings. The join must go through the full Parquet DataFrames (`data["entities"]`, `data["relationships"]`) to map between the two ID spaces.

**Logic:**

1. For each report row in `context["reports"]`:
   a. Match the context report back to `data["community_reports"]` by `human_readable_id` (the `"id"` column in context corresponds to the `human_readable_id` column in the Parquet)
   b. Read the `community` field (integer) from the matched Parquet row
   c. Look up that community in `data["communities"]` to get `entity_ids` (UUID list), `relationship_ids` (UUID list), `text_unit_ids` (UUID list)
   d. Filter `context["entities"]` to those whose title matches entities in the community (join context `"id"` -> `data["entities"]["human_readable_id"]` -> check if `data["entities"]["id"]` is in the community's `entity_ids`)
   e. Same join pattern for `context["relationships"]`
   f. Same for `context["sources"]` (text units) — join context `"id"` -> `data["text_units"]["human_readable_id"]` -> check if `data["text_units"]["id"]` is in the community's `text_unit_ids`
   g. Same for `context.get("claims")` (covariates, if present)
   h. For each entity/relationship, resolve document provenance: entity/relationship row in full Parquet -> `text_unit_ids` (UUID list) -> `data["text_units"]` rows -> `document_id` (singular string field) -> `data["documents"]` row -> `title` (hash suffix stripped)
   i. For each text_unit, resolve document provenance: join context `"id"` back to `data["text_units"]` by `human_readable_id` -> `document_id` -> `data["documents"]`

2. For **global search** (reports only, no entities/relationships/text_units in context): each provenance entry has the report content but empty entity/relationship/text_unit/covariate lists.

3. For **basic search** (no community reports): create a single provenance entry with empty report fields and populate `text_units` from the context `"sources"` key. Resolve document provenance by joining the context `"id"` (human_readable_id) back to `data["text_units"]` to get `document_id`.

4. Return the list of provenance entries.

**Document title cleaning:** Strip the `_[0-9a-f]{8}` hash suffix from document titles.

**Document ID field:** The `text_units` Parquet uses `document_id` (singular string, not a list). Handle `None` values gracefully.

### Backend Integration: `app/services/graphrag_service.py`

Each of the four search functions (`local_search`, `global_search`, `drift_search`, `basic_search`) changes its return:

**Before:**
```python
from app.services.graphrag_citations import process_citations
clean_response, sources = process_citations(response, data, "graphrag_local")
return {
    "response": clean_response,
    "sources": sources,
    "context": _serialize_context(context),
}
```

**After:**
```python
from app.services.graphrag_provenance import build_provenance
provenance = build_provenance(context, data, "graphrag_local")
return {
    "response": response,
    "provenance": provenance,
    "context": _serialize_context(context),
}
```

`_load_search_data()` already loads `"communities"` and `"documents"` — no change needed there.

**Think-tag stripping:** The current `process_citations` strips `<think>`/`<thinking>` tags from the response. This must be preserved in `graphrag_service.py` before returning the response text, so tags from Ollama thinking models don't leak into `content_text`. Add a `_strip_think_tags()` utility to `graphrag_service.py` and apply it to `response` before returning.

### Task Serialization: `app/workers/graphrag_tasks.py`

**Before:**
```python
result_item = {
    "score": 1.0,
    "modality": "graphrag_response",
    "content_text": graphrag_result.get("response", response),
    "classification": "UNCLASSIFIED",
    "context": {
        "source": strategy,
        "sources": graphrag_result.get("sources", []),
        "graphrag_context": graphrag_result.get("context", {}),
    },
}
```

**After:**
```python
result_item = {
    "score": 1.0,
    "modality": "graphrag_response",
    "content_text": response,
    "classification": "UNCLASSIFIED",
    "context": {
        "source": strategy,
        "provenance": graphrag_result.get("provenance", []),
        "graphrag_context": graphrag_result.get("context", {}),
    },
}
```

### Frontend: `frontend/src/components/QueryPage.tsx`

**Remove:**
- `CitationLink` component
- `SourcesPanel` component and `SourceEntry` interface
- `renderWithCitations` helper
- All `[n]` citation marker rendering in preview and full text sections
- `sources` and `hasSources` extraction from context

**Add:** `ProvenancePanel` component with collapsible hierarchy:

```
▼ Provenance (3 community reports)
  ▼ SA-2 Guideline & Fan Song Radar Air-Defense Community
    ▼ Report Content
      (full_content rendered, could be markdown)
    ▼ Entities (12)
      FAN SONG · FIRE_CONTROL_SYSTEM — Engagement radar...
        📄 Red SAM
    ▼ Relationships (8)
      S-75 DVINA → V-750 — Interceptor component
        📄 Red SAM
    ▼ Source Texts (5)
      "The SA-2 Guideline uses command guidance..."
        📄 Red SAM
    ▼ Covariates (0)
```

**Behavior:**
- Top-level "Provenance" collapsed by default, shows report count
- Each community report collapsible, title always visible
- Sub-sections (entities, relationships, text_units, covariates) each collapsible with counts
- Source documents shown as inline badges under each item
- Text chunks truncated to ~300 chars, expandable on click

**Revert:** Preview and full text rendering back to plain text (no `renderWithCitations` wrapping).

### CSS: `frontend/src/styles.css`

**Remove:** `.citation-link`, `.citation-link:hover`, `.source-entry:target`

**Add:** Minimal styles for provenance panel (collapsible sections, document badges, indentation).

### Prompt Changes: `app/services/graphrag_prompts.py`

**Revert** the citation instruction blocks added to:
- `get_local_search_prompt()`
- `get_global_search_reduce_prompt()`
- `get_drift_search_prompt()`
- `get_basic_search_prompt()`

Remove the `IMPORTANT — Inline Citations:` blocks. The prompts return to their original state.

### Verification Checklist: `VERIFICATION_CHECKLIST.md`

Update the GraphRAG citation provenance entry (added as item 2.32) to reflect the context-based approach:

```
| GraphRAG context provenance (all 4 search types) | GraphRAG responses lack source traceability | Run GraphRAG Local query; provenance array contains community reports with entities + source documents | 2.32 |
```

Update Known Fragile Features item 14:

```
14. **GraphRAG context provenance** (2.32) — Depends on communities.parquet having entity_ids/relationship_ids. Test all 4 strategies; verify provenance array populated with source_documents.
```

## Files Changed

| Action | File | What |
|--------|------|------|
| Delete | `app/services/graphrag_citations.py` | Remove LLM citation parser entirely |
| Delete | `tests/unit/test_graphrag_citations.py` | Remove citation parser tests |
| Create | `app/services/graphrag_provenance.py` | Context-based provenance builder |
| Create | `tests/unit/test_graphrag_provenance.py` | Provenance builder tests |
| Modify | `app/services/graphrag_prompts.py` | Revert citation instruction additions |
| Modify | `app/services/graphrag_service.py` | Wire provenance builder, add think-tag stripping |
| Modify | `app/workers/graphrag_tasks.py` | Replace sources with provenance in result |
| Modify | `frontend/src/components/QueryPage.tsx` | Replace citation components with ProvenancePanel |
| Modify | `frontend/src/styles.css` | Replace citation CSS with provenance CSS |
| Modify | `VERIFICATION_CHECKLIST.md` | Update checklist entry |

## Edge Cases

- **No community reports** (basic search): Single provenance entry with empty report fields, text_units populated from context `"sources"` key.
- **Global search** (reports only): Provenance entries have report_content but empty entity/relationship/text_unit/covariate lists because the global context builder only includes reports.
- **Community not found in communities.parquet**: Log warning, include report with empty entity/relationship/text_unit lists. The report content itself is still valuable.
- **Report not matched back to Parquet**: If the context report's `human_readable_id` doesn't match any row in `data["community_reports"]`, skip the community join and include the report with empty member lists.
- **Entity belongs to multiple communities**: Entity appears under each community report it belongs to. Duplication is acceptable — each community provides different analytical context.
- **Empty context** (search returns no results): `provenance` is an empty array `[]`.
- **null document_id on text_units**: The `document_id` field is singular (string or None). Skip document resolution when None; `source_documents` will be empty. Don't crash.
- **Document title hash suffix**: Strip `_[0-9a-f]{8}$` pattern from document titles for display.
- **Think tags in response**: Strip `<think>`/`<thinking>` tags from response text before returning, preserving current behavior.
- **Context dict key variations**: Use defensive `.get()` with empty DataFrame fallbacks for all context keys. Log warnings for unexpected context structure.
