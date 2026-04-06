# Design: DoclingDocument as Authoritative Mutable Artifact (v2)

**Date:** 2026-04-06
**Status:** Revised after review
**TODO items:** #60, #61, #62
**Scope:** This spec covers the authoritative DoclingDocument artifact, native chunking, and native enrichment. It does NOT cover Docling-Graph template alignment (#63), provenance preservation (#64), or ArcadeDB-first retrieval (#65) — those are separate specs.

---

## Problem

The DoclingDocument JSON is persisted once in `prepare_document` and never updated. Translation and picture descriptions mutate DocumentElement rows but not the DoclingDocument. The JSON in MinIO is permanently stale.

## Review Findings Addressed

1. **No `element_uid` in native Docling JSON** — Use `self_ref` (e.g., `#/texts/0`) as the native join key. Build a `self_ref → element_uid` mapping during `prepare_document` and persist it.
2. **Custom mutation doesn't affect native text fields** — Translate by mutating the canonical `.text` field on items (what HybridChunker and Docling-Graph read), preserving originals in a separate `_original_text` overlay.
3. **Viewer hover-tooltip contract** — Image descriptions persist as `pictures[].annotations[{kind: "description"}]` matching the `<docling-tooltip>` contract.
4. **Translation toggle** — Original text stored in `_enrichments.translations` overlay keyed by `self_ref`. The viewer `/element-translations` endpoint reads from this overlay. Native tools see the translated `.text` field.
5. **HybridChunker output mapping** — Detailed rules for chunk_id, artifact_id, page_number, modality, and the image_description secondary pass.
6. **`/docling-raw` endpoint changed behavior** — Remove runtime annotation injection since enrichments are now in the persisted JSON. Document as a behavior change.
7. **HybridChunker tokenizer API** — Use tokenizer object, not deprecated string form.
8. **MinIO synchronization** — Postgres updates and MinIO writes happen atomically within each stage's DB transaction boundary. On retry, both are rewritten.
9. **Not a complete native-first fix** — Explicitly scoped. Regex mention building and custom Docling-Graph normalization remain for separate specs.

---

## Design

### Identity bridge: `self_ref` ↔ `element_uid`

Native Docling items have `self_ref` (e.g., `#/texts/0`, `#/pictures/1`). Our pipeline uses synthetic `element_uid`. These are bridged:

**In `prepare_document`:** During `_extract_elements`, capture each item's `self_ref` alongside the synthetic `element_uid`. Persist a mapping as `_enrichments.identity_map` in the DoclingDocument JSON:

```json
{
  "_enrichments": {
    "identity_map": {
      "#/texts/0": "1-0-text-a1b2c3",
      "#/pictures/1": "2-3-image-d4e5f6"
    }
  },
  "texts": [...],
  "pictures": [...]
}
```

Also store `self_ref` on the DocumentElement row (new column or in `element_metadata` JSONB) so stages can look up the native path.

**In `converter.py`:** During `_extract_elements`, extract `self_ref` from each DocItem and include it in the ConvertedElement metadata.

### Translation enrichment (`detect_and_translate`)

**Native text replacement:** Mutate the canonical `.text` field on each translated item so native tools (HybridChunker, Docling-Graph) see translated content:

```python
doc_dict = json.loads(download_from_minio(...))

# Save originals before mutation
translations = {}
for element_uid, translated_text in translated_elements.items():
    self_ref = identity_map_reverse[element_uid]  # element_uid → self_ref
    collection, idx = parse_self_ref(self_ref)     # "#/texts/0" → ("texts", 0)
    item = doc_dict[collection][idx]
    translations[self_ref] = {
        "original_text": item["text"],
        "translated_text": translated_text,
    }
    item["text"] = translated_text  # Native tools now see translated text

doc_dict.setdefault("_enrichments", {})["translations"] = translations
upload_to_minio(doc_dict)
```

**Viewer contract preserved:** The `/element-translations` endpoint reads from `_enrichments.translations` to serve the toggle UI. Original text is always available there.

### Picture description enrichment (`derive_picture_descriptions`)

**Native annotation pattern:** Match the `<docling-tooltip>` contract by persisting descriptions as annotations:

```python
for pic_idx, description in descriptions.items():
    pic = doc_dict["pictures"][pic_idx]
    # Update the canonical text field for native tools
    pic["text"] = description
    # Persist as annotation for the viewer hover contract
    if "annotations" not in pic:
        pic["annotations"] = []
    pic["annotations"].append({
        "kind": "description",
        "text": description,
        "source": "llm",
        "model": model_name,
    })

upload_to_minio(doc_dict)
```

### `/docling-raw` endpoint behavior change

**Before:** Loads raw JSON from MinIO, then injects image-description annotations at read time from Postgres.

**After:** Returns the enriched JSON directly from MinIO. Remove the runtime annotation injection code since annotations are now persisted in the JSON. This is a behavior change — document it and update the endpoint.

### Native chunking (`derive_text_chunks_and_embeddings`)

**Reconstruct DoclingDocument, chunk natively:**

```python
from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument
from transformers import AutoTokenizer

doc_dict = json.loads(download_from_minio(...))
doc = DoclingDocument.model_validate(doc_dict)

tokenizer = AutoTokenizer.from_pretrained(settings.text_embedding_model)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=settings.chunk_max_tokens,
)

for chunk in chunker.chunk(doc):
    chunk_text = chunk.text
    # chunk.meta.doc_items → list of source DocItems with self_ref
    # chunk.meta.headings → heading hierarchy
```

**Chunk schema mapping rules:**

| Field | Source |
|-------|--------|
| `chunk_id` | Deterministic: `sha256(document_id + self_refs + chunk_index)` → UUID |
| `document_id` | From pipeline context |
| `artifact_id` | Resolve via `self_ref → element_uid → artifact` mapping |
| `page_number` | From `chunk.meta.doc_items[0].prov[0].page_no` (native Docling provenance) |
| `modality` | Infer from source item labels: `section_header`/`paragraph` → "text", `table` → "table", etc. |
| `classification` | From document metadata |
| `chunk_text` | `chunk.text` (already translated if translation stage ran) |

**Image description secondary pass:** After main chunking, scan `pictures` in the DoclingDocument for items with `annotations[kind=description]`. Split long descriptions into sections (same logic as current `split_description_sections`). Create TextChunk rows with `modality=image_description` and bidirectional `SAME_ARTIFACT` links. This preserves the current hybrid retrieval contract.

**Fallback:** If DoclingDocument reconstruction fails (pre-refactor documents, corrupt JSON), fall back to current `structure_aware_chunk()` over DocumentElement rows.

### Graph extraction (`derive_ontology_graph`)

Load the enriched DoclingDocument JSON from MinIO and pass it directly to the Docling-Graph service. The JSON now contains translated text (in canonical `.text` fields) and picture descriptions (in `pictures[].text` + annotations). Remove the `_enriched_text` reconstruction hack.

### MinIO synchronization

Each enrichment stage follows this order:
1. Load DoclingDocument JSON from MinIO
2. Apply enrichments to the dict
3. Update DocumentElement rows in Postgres (backward compat)
4. Re-persist DoclingDocument JSON to MinIO
5. Commit Postgres transaction

On Celery retry, steps 1-5 re-execute fully — both Postgres and MinIO are overwritten. MinIO writes are idempotent (same key, full replacement). The DoclingDocument JSON is the authoritative state; DocumentElement rows are a derived relational index.

### Files changed

| File | Change |
|------|--------|
| `docker/docling/app/converter.py` | Extract `self_ref` from each DocItem in `_extract_elements`, include in ConvertedElement metadata |
| `app/services/docling_client.py` | Pass `self_ref` through in ExtractedChunk metadata |
| `app/workers/pipeline.py` (prepare) | Build and persist `_enrichments.identity_map` in DoclingDocument JSON |
| `app/workers/pipeline.py` (translate) | Reload JSON, mutate canonical `.text` fields, persist `_enrichments.translations`, re-upload |
| `app/workers/pipeline.py` (pictures) | Reload JSON, add annotations + update `.text`, re-upload |
| `app/workers/pipeline.py` (chunks) | Load enriched JSON, reconstruct DoclingDocument, use HybridChunker |
| `app/workers/pipeline.py` (graph) | Load enriched JSON directly, remove _enriched_text hack |
| `app/services/chunking.py` | Keep as fallback; no changes |
| `app/api/v1/sources.py` (docling-raw) | Remove runtime annotation injection; return MinIO JSON directly |
| `app/api/v1/sources.py` (translations) | Read from `_enrichments.translations` in MinIO JSON |

### Not changed

- DocumentElement table schema — still populated for relational queries
- Retrieval backfill functions — still read from Postgres chunks
- Docling-Graph normalization — separate spec (#63, #64)
- Mention building — separate spec (#64)
- ArcadeDB structural edges — separate spec (#65)
