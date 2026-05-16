# Document-metadata pass — extension proposal

**Status:** design — not yet implemented.
**Related:** [sa2_extraction_runs.md](./sa2_extraction_runs.md) (run history), `app/services/document_analysis.py` (existing pass), `app/services/table_normalization/_pipeline_hooks.py` (current Option-A regex workaround).

## What exists today

`app/services/document_analysis.py::extract_document_metadata(markdown, classification_text=None)` runs once per document during ingest (called from `app/workers/pipeline.py:4784-4785`). Output persists to `Document.document_metadata` (Postgres JSONB) and to the ArcadeDB Document vertex.

**Architecture:** 4 parallel LLM calls via `ThreadPoolExecutor(max_workers=4)` — independent reads of the same markdown.

| current field | prompt source (in `app/config.py`) | downstream consumer |
|---|---|---|
| `document_summary` | `doc_analysis_summary_prompt` | fed into `describe_pictures()` so picture captions know doc context |
| `date_of_information` | `doc_analysis_date_prompt` | written to ArcadeDB Document vertex; reconciled with `created_at` |
| `source_characterization` | `doc_analysis_source_prompt` | written to ArcadeDB Document vertex (formatted Organization / Type / Reliability) |
| `classification` | `doc_analysis_classification_prompt` | normalized to one of `UNCLASSIFIED / CUI / FOUO / SECRET / TOP SECRET`; written to ArcadeDB |

**Model:** configurable via `doc_analysis_llm_model`; runs at `temperature=0.1`, `think` per `get_doc_analysis_llm_think()`, timeout per `doc_analysis_timeout`. Input capped at `ollama_num_ctx * 3` chars (~168K default).

Output shape persisted as Postgres JSONB on `Document.document_metadata`:
```json
{
  "document_summary": "...",
  "date_of_information": "Mon Jan 27 11:18:09 UTC 2014",
  "classification": "UNCLASSIFIED",
  "source_characterization": "Organization: ...\nType: ...\nReliability: 8/10",
  "generated_at": "2026-05-15T22:18:09+00:00"
}
```

## Extension — three new fields

Add three fields to the metadata pass. All three are LLM-extracted with the same parallel-fan-out infrastructure already in place — drop in three more `ThreadPoolExecutor` futures alongside the existing four.

### 1. `document_origin`

Origin of the document's subject matter — used downstream to pick unit convention defaults and inform extraction prompts about domain conventions.

- **Prompt sketch** (new setting `doc_analysis_origin_prompt`):
  > "Identify the originating nation, military service, or organizational tradition of the systems/equipment described in this document. Return ONE of: `SOVIET`, `RUSSIAN`, `US_DOD`, `US_NAVY`, `US_AIR_FORCE`, `UK_MOD`, `EUROPEAN_NATO`, `CHINESE`, `IRANIAN`, `ISRAELI`, `OTHER`, `UNKNOWN`. Return only the token."
- **Validation:** normalize to one of the allowed enum values; default to `UNKNOWN` on parse failure (same pattern as `classification`).
- **Persisted as:** `document_metadata.document_origin`

### 2. `unit_convention`

Document-level unit convention — replaces the per-table regex heuristic currently in `_pipeline_hooks.detect_unit_convention()`.

- **Prompt sketch** (new setting `doc_analysis_unit_convention_prompt`):
  > "What system of measurement does this document use for numeric specifications? Return EXACTLY one JSON object with these fields:
  > - `primary`: one of `metric`, `imperial`, `mixed`
  > - `notes`: one short sentence explaining the evidence (e.g. 'Soviet origin; all ranges in km and altitudes in m', or 'US Air Force tech order; altitudes in feet, ranges in nautical miles')
  > Return ONLY the JSON object."
- **Validation:** parse JSON; verify `primary` is one of the three allowed tokens; default to `{"primary": "metric", "notes": "default — origin metric per modern technical convention"}` on parse failure.
- **Persisted as:** `document_metadata.unit_convention` (nested object)

### 3. `entity_hints`

Canonical entity roster the document expects — feeds the cross-pass identity-alignment roster (Step 5 of the original handoff).

- **Prompt sketch** (new setting `doc_analysis_entity_hints_prompt`):
  > "Identify the primary named systems, vehicles, weapons, sensors, or platforms described in this document. Return EXACTLY one JSON object with these fields:
  > - `systems`: array of `{ canonical_name, aliases: [], system_type }` where `system_type` is one of `MISSILE`, `RADAR`, `AIRCRAFT`, `SHIP`, `VEHICLE`, `SENSOR`, `WEAPON`, `OTHER`.
  > - Up to 10 entries; rank by document prominence.
  > Return ONLY the JSON object."
- **Validation:** parse JSON; cap to 10 entries; drop entries missing `canonical_name` or with unrecognized `system_type`. Default to `{"systems": []}` on failure.
- **Persisted as:** `document_metadata.entity_hints` (nested object)

Example post-extension `document_metadata` for SA-2:
```json
{
  "document_summary": "Technical reference for the Soviet S-75 / SA-2 Guideline ...",
  "date_of_information": "Mon Jan 27 11:18:09 UTC 2014",
  "classification": "UNCLASSIFIED",
  "source_characterization": "Organization: APA\nType: technical report\nReliability: 8/10",
  "document_origin": "SOVIET",
  "unit_convention": {
    "primary": "metric",
    "notes": "Soviet-origin SAM reference; ranges and altitudes in metres"
  },
  "entity_hints": {
    "systems": [
      {"canonical_name": "S-75", "aliases": ["SA-2", "Guideline", "Dvina"], "system_type": "MISSILE"},
      {"canonical_name": "Fan Song", "aliases": ["SNR-75", "RSN-75"], "system_type": "RADAR"},
      {"canonical_name": "HQ-2", "aliases": ["HQ-1"], "system_type": "MISSILE"}
    ]
  },
  "generated_at": "2026-05-15T22:18:09+00:00"
}
```

## How the schema passes consume it

The `/extract-pass` endpoint (`docker/docling-graph/app/main.py`) currently receives `docling_document_json`, `bundle_key`, `pass_name`, and optionally `upstream_entities`. Add an optional `document_metadata` field:

```python
# pipeline.py — when building the extract-pass request body
request_body = {
    "bundle_key": bundle_key,
    "pass_name": pass_def.name,
    "document_id": document_id,
    "docling_document_json": doc_json,
    "document_metadata": document.document_metadata,   # NEW — pass the full record
    "upstream_entities": upstream_entities or [],
}
```

Inside `main.py`, two consumption points:

### Consumption A — `unit_convention` → synth-block UNIT_HINT preamble

Replaces the regex-based `detect_unit_convention(table_idx, doc_json)` helper in `_pipeline_hooks.py`. The render path becomes:

```python
# main.py — when building synth chunks
_unit_convention = (
    (document_metadata or {})
    .get("unit_convention", {})
    .get("primary", "metric")
)
# fall back to the regex detector if no metadata pass output is present
if _unit_convention == "metric":
    _unit_hint = UNIT_HINT_METRIC
elif _unit_convention == "imperial":
    _unit_hint = UNIT_HINT_IMPERIAL
else:
    _unit_hint = UNIT_HINT_MIXED  # explicit "respect labeled units per row"
```

Then `render_for_graph(nt, ..., unit_hint=_unit_hint)` threads through.

### Consumption B — `entity_hints` → identity roster preamble

For non-identity passes (kinematics, airframe, propulsion, etc.), prepend a compact roster to the user prompt:

```
=== KNOWN ENTITY ROSTER FOR IDENTITY RESOLUTION ONLY ===
This roster is NOT evidence. Do not emit an entity or property solely because it appears here.
Use it only to map aliases/designations in the current batch to canonical system_name values.

MISSILE_SYSTEM:
- canonical: S-75 | aliases: SA-2, Guideline, Dvina
- canonical: HQ-2 | aliases: HQ-1

RADAR_SYSTEM:
- canonical: Fan Song | aliases: SNR-75, RSN-75
=== END KNOWN ENTITY ROSTER ===
```

Built in `main.py` from `document_metadata.entity_hints.systems` filtered to the entity type the pass cares about (MISSILE for missile_* passes, RADAR for radar_* passes). Identity passes (`missile_identity`, `radar_identity`) do NOT receive the roster — they're the authoritative source.

The roster construction goes in a small helper, e.g. `_build_identity_roster(pass_name, document_metadata) -> str`. Token cap (e.g. 500 chars) prevents bloat.

## Wiring changes

| file | change |
|---|---|
| `app/config.py` | add 3 prompt settings: `doc_analysis_origin_prompt`, `doc_analysis_unit_convention_prompt`, `doc_analysis_entity_hints_prompt` |
| `app/services/document_analysis.py` | add 3 prompts to the parallel-fan-out dict; parse JSON for unit_convention + entity_hints; validation + defaults |
| `app/workers/pipeline.py::_build_extract_pass_body` | include `document_metadata` in the request body |
| `docker/docling-graph/app/main.py` | (a) read `document_metadata` from request body; (b) thread `unit_convention` into render_for_graph; (c) thread `entity_hints` into the prompt builder for non-identity passes |
| `app/services/table_normalization/_pipeline_hooks.py` | keep `detect_unit_convention()` as a fallback when metadata is absent (legacy docs or pre-extension ingests) |
| `docker/docling-graph/repo/.../delta_batch_prompt.py` (or equivalent) | accept and render the identity roster preamble in the user prompt |

## Test plan

- Unit tests on `extract_document_metadata`: mock LLM responses for each new field; verify validation + defaults
- Unit test on roster builder: given a metadata record, verify the rendered preamble respects type filter + token cap
- Integration test on `/extract-pass`: with `document_metadata` set, verify the request prompt contains the expected UNIT_HINT + roster; with metadata absent, verify fallback to current behavior

## Storage & re-ingestion

- New fields persist into existing `Document.document_metadata` JSONB column — no schema migration needed
- For existing documents ingested BEFORE this extension lands, a backfill script can call `extract_document_metadata(markdown)` again and patch only the missing fields (`document_origin`, `unit_convention`, `entity_hints`). Existing fields would not be overwritten unless explicitly requested

## Decision-relative trade-offs

- **Why a metadata-pass extension vs an LLM call per `/extract-pass`?** The metadata pass runs ONCE per document; extraction passes run N times. Metadata-pass extension amortizes LLM cost.
- **Why not just regex-detect unit convention from caption / prose?** That's what the current `_pipeline_hooks.detect_unit_convention()` does. It works for explicit cases (caption says "(metric)") but misses implicit cases (Soviet-origin doc with no unit declaration). LLM-based origin classification handles implicit cases.
- **Why allow the regex detector to stay as fallback?** Backward compatibility with documents ingested before this extension exists. The fallback is also cheaper than a forced re-ingest for an entire corpus.
- **Why cap `entity_hints.systems` at 10?** Roster bloat hurts every extraction prompt. Top-10 prominent systems is enough for SA-2-scale specs; if a doc legitimately has 50 systems, the truncation forces us to keep the 10 most prominent and the extraction LLM rediscovers the rest from the body.

## What this does NOT solve

- **Per-table classification** (telling the kinematics pass "skip the radar spec table") — still done by `is_table_relevant_for_pass()` row-label-alias matching. Could later move into `document_metadata.table_inventory` if needed.
- **Per-pass canonical-name agreement** — `entity_hints` helps identity resolution but doesn't enforce that all passes emit the same canonical name; that requires merge-layer dedup work.
- **Cross-document entity dedup** — `entity_hints` is per-document; making it corpus-level requires a separate ontology layer.
