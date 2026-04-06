# Design: DoclingDocument as Authoritative Mutable Artifact (v4)

**Date:** 2026-04-06
**Status:** Revised after third review
**TODO items:** #60, #61, #62

---

## Key Changes from v3

1. **Picture descriptions** use native `PictureMeta.description` (a `DescriptionMetaField` with `text`, `confidence`, `created_by`). Legacy `annotations` mirrored for viewer hover contract.
2. **Markdown semantics preserved:** `docling_document.md` stays original. `docling_document_translated.md` stays translated. No semantic change to existing endpoints.
3. **Docling-Graph wrapper updated** to extract `_enrichments.context` and prepend to the document before calling `run_pipeline()`. No silent context loss.
4. **Complete file list** of all markdown consumers/cleaners that need updates.

---

## Design

### Identity bridge

`self_ref` (native Docling, e.g., `#/texts/0`) ↔ `element_uid` (pipeline-synthetic). Bridged via `_enrichments.identity_map` in the JSON, built during `prepare_document`. `self_ref` also stored in `DocumentElement.element_metadata` JSONB.

### Cross-store consistency

Last-writer-wins with full-stage idempotence. `_enrichments.version` counter tracks mutations. On retry, full stage re-executes (Postgres upserts + MinIO upload are both idempotent).

### Translation enrichment

**Canonical `.text` mutated** on each item so HybridChunker and Docling-Graph see translated content. Originals preserved in `_enrichments.translations[self_ref] = {original_text, translated_text, language}`.

**Markdown artifact contract (v4 decisions):**

- `docling_document.md` — original Docling markdown, written in `prepare_document`. Later modified by `derive_picture_descriptions` which appends a description appendix (current behavior, kept). This is the file served by the `/docling` markdown endpoint and the DoclingViewer. It is NOT immutable — it accumulates picture descriptions. Classification detection in `derive_document_metadata` reads this file BEFORE picture descriptions run (stages are ordered: metadata runs before pictures in the DAG), so classification sees unmodified original text.

- `docling_document_translated.md` — **regenerated from the authoritative DoclingDocument** after translation mutates `.text` fields. Replaces the current custom concatenation from DocumentElement rows with `DoclingDocument.model_validate(doc_dict).export_to_markdown()`. This keeps the translated markdown in sync with the authoritative JSON.

- `docling_document.json` — the authoritative enriched JSON. Mutations accumulate here across stages.

This preserves: `/docling` endpoint serves the original+pictures markdown, `/translation` endpoint serves translated markdown (now derived from the authoritative JSON instead of DocumentElement rows), `derive_document_metadata` reads original markdown for classification and translated markdown for summary/date/source.

### Picture descriptions

**Native-first:** Write to `PictureMeta.description` (the canonical field):
```python
pic_item["meta"] = pic_item.get("meta") or {}
pic_item["meta"]["description"] = {
    "text": description_text,
    "confidence": None,
    "created_by": f"llm:{model_name}",
}
```

**Legacy mirror for viewer:** Also write to `annotations[]` with `DescriptionAnnotation` shape so the existing `<docling-tooltip>` hover UI works until it's updated to read from `meta.description`.

### Docling-Graph context injection

The Docling-Graph wrapper injects document summary/classification as a synthetic text node at the beginning of the DoclingDocument before `run_pipeline()`. This uses the native `DoclingDocument.add_text()` API on a temp copy:

```python
def run_extraction_pipeline(docling_document_json, templates, unified_template):
    import copy
    doc_json = copy.deepcopy(docling_document_json)

    # Inject summary/classification context as a leading text node
    enrichments = doc_json.pop("_enrichments", {})
    context = enrichments.get("context", {})
    if context.get("summary") or context.get("classification"):
        doc = DoclingDocument.model_validate(doc_json)
        context_text = ""
        if context.get("summary"):
            context_text += f"[Document Summary]: {context['summary']}\n"
        if context.get("classification"):
            context_text += f"[Classification]: {context['classification']}\n"
        if context_text:
            doc.add_text(text=context_text, label="paragraph")
        # Re-serialize to dict for temp file
        doc_json = doc.export_to_dict()

    with tempfile.NamedTemporaryFile(...) as tmp:
        json.dump(doc_json, tmp)
        config = build_pipeline_config(source=tmp.name, ...)
        return run_pipeline(config)
```

This replaces the `_enriched_text` hack. The context node is added to a deep copy so the persisted MinIO JSON is not mutated.

### HybridChunker

```python
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

tok = AutoTokenizer.from_pretrained(settings.text_embedding_model)
hf_tokenizer = HuggingFaceTokenizer(tokenizer=tok, max_tokens=settings.chunk_max_tokens)
chunker = HybridChunker(tokenizer=hf_tokenizer)
```

### Chunk schema mapping

| Field | Source |
|-------|--------|
| `chunk_id` | `UUID(md5(sha256(doc_id:self_refs:chunk_idx:model_version)))` |
| `artifact_id` | Via `self_ref → identity_map → element_uid → Artifact` |
| `page_number` | `chunk.meta.doc_items[0].prov[0].page_no` |
| `modality` | From source item label |
| `chunk_text` | `chunk.text` (translated if translation ran) |

Image description secondary pass: scan `pictures` for `meta.description`, split sections, create `image_description` TextChunks with `SAME_ARTIFACT` links.

Fallback: `structure_aware_chunk()` over DocumentElement rows if DoclingDocument reconstruction fails.

### `/docling-raw` endpoint

Returns the enriched JSON from MinIO directly. **Remove runtime annotation injection** (currently in `sources.py` lines ~817-836) since descriptions are now in the persisted JSON.

### `/element-translations` endpoint

Reads from `_enrichments.translations` in MinIO JSON, reverse-maps `self_ref → element_uid` via `identity_map`. Response shape unchanged: `{element_uid, original_text, translated_text}`.

**Backward compat for pre-v4 documents:** If `_enrichments.translations` is missing from the JSON, fall back to reading from Postgres `DocumentElement.translated_text` (current behavior). Same for `identity_map` — if missing, the endpoint falls back to the Postgres path.

### Backward compatibility for pre-v4 documents

Documents ingested before this refactor will not have `_enrichments`, `identity_map`, or native `meta.description` in their JSON. All new code paths include explicit fallbacks:

| Feature | v4 path | Pre-v4 fallback |
|---------|---------|-----------------|
| `/element-translations` | `_enrichments.translations` + `identity_map` | Postgres `DocumentElement.translated_text` |
| `/docling-raw` | Return enriched JSON directly | Return raw JSON + runtime annotation injection (current code kept behind version check) |
| HybridChunker | Load JSON, reconstruct DoclingDocument | `structure_aware_chunk()` over DocumentElement rows |
| Graph extraction | Load enriched JSON with `_enrichments.context` | Reconstruct `_enriched_text` from DocumentElement rows |
| Chunk `artifact_id` resolution | `identity_map` → `element_uid` → Artifact | Fall back to `artifact_id_to_element_uid` map built from DocumentElement rows |

The version check is simple: `doc_dict.get("_enrichments", {}).get("version")`. If absent, the document is pre-v4 and all fallback paths activate.

### Graph extraction fallback

If DoclingDocument JSON missing/corrupt: reconstruct enriched text from DocumentElement rows, wrap as `{"_enriched_text": text}`, log warning.

---

## Complete file change list

### Pipeline stages
| File:location | Change |
|---------------|--------|
| `docker/docling/app/converter.py` `_extract_elements` | Extract `self_ref` from each DocItem, include in ConvertedElement metadata |
| `app/services/docling_client.py` | Pass `self_ref` through in ExtractedChunk metadata |
| `app/workers/pipeline.py` `prepare_document` (~L815) | Build `_enrichments.identity_map`, store `self_ref` in `element_metadata` JSONB |
| `app/workers/pipeline.py` `detect_and_translate` (~L1276) | Reload JSON, mutate `.text`, persist `_enrichments.translations`, re-upload JSON. Regenerate `docling_document_translated.md` from the mutated DoclingDocument via `export_to_markdown()` (replaces custom DocumentElement concatenation) |
| `app/workers/pipeline.py` `derive_picture_descriptions` (~L1479) | Reload JSON, add `meta.description` + legacy `annotations`, re-upload JSON |
| `app/workers/pipeline.py` `derive_text_chunks_and_embeddings` (~L1673) | Load JSON, reconstruct DoclingDocument, use HybridChunker. Fallback to `structure_aware_chunk()` |
| `app/workers/pipeline.py` `derive_ontology_graph` (~L2234) | Load enriched JSON, add `_enrichments.context`, pass to service. Remove `_enriched_text` reconstruction |

### Docling-Graph wrapper
| File:location | Change |
|---------------|--------|
| `docker/docling-graph/app/main.py` `run_extraction_pipeline` (~L88) | Extract `_enrichments.context`, inject summary/classification before `run_pipeline()` |

### API endpoints (markdown consumers)
| File:location | Current behavior | Change |
|---------------|-----------------|--------|
| `app/api/v1/sources.py` `get_docling_document` (~L659) | Serves `docling_document.md` | No change — file is original + picture appendix (current behavior) |
| `app/api/v1/sources.py` `get_docling_raw` (~L818) | Loads JSON + injects annotations at runtime | Remove runtime injection for v4 documents (check `_enrichments.version`); keep injection for pre-v4 |
| `app/api/v1/sources.py` `get_translation` (~L882) | Reads `docling_document_translated.md` | No change — file now regenerated from authoritative DoclingDocument instead of DocumentElement rows |
| `app/api/v1/sources.py` `get_element_translations` (~L916) | Reads from Postgres DocumentElement | Read from `_enrichments.translations` with fallback to Postgres for pre-v4 documents |
| `app/api/v1/sources.py` `_hard_delete_document` (~L504) | Deletes `docling_document.md`, `.json`, `_translated.md` | No change — same files still exist |

### Pipeline markdown readers
| File:location | Current behavior | Change |
|---------------|-----------------|--------|
| `app/workers/pipeline.py` `derive_document_metadata` (~L1104) | Reads `docling_document.md` as original | No change — file stays original |
| `app/workers/pipeline.py` `derive_document_metadata` (~L1115) | Reads `docling_document_translated.md` | No change — file now regenerated from authoritative JSON |
| `app/workers/pipeline.py` `derive_picture_descriptions` (~L1406) | Reads `docling_document.json` | Now reads enriched JSON (has translations) |
| `app/workers/pipeline.py` `derive_picture_descriptions` (~L1496) | Reads/writes `docling_document.md` | Appends description appendix to original markdown (current behavior unchanged) |

### Chunking
| File | Change |
|------|--------|
| `app/services/chunking.py` | Keep as fallback. No changes. |
