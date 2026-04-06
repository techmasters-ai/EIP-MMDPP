# Design: DoclingDocument as Authoritative Mutable Artifact (v3)

**Date:** 2026-04-06
**Status:** Revised after second review
**TODO items:** #60, #61, #62
**Scope:** Authoritative DoclingDocument, native chunking, enrichment. NOT Docling-Graph templates (#63), provenance (#64), or ArcadeDB-first retrieval (#65).

---

## Problem

DoclingDocument JSON persists once, never updates. Translation and picture descriptions mutate DocumentElement rows but not the JSON. Downstream stages and the API serve stale data.

## Key Design Decisions

### 1. Cross-store consistency model (Finding 1)

MinIO and Postgres are separate stores with no distributed transaction. The consistency model is **last-writer-wins with full-stage idempotence**:

- Each stage writes Postgres THEN MinIO. If MinIO fails after Postgres commits, the Celery retry re-executes the full stage — Postgres rows are re-upserted (idempotent), MinIO is re-uploaded (idempotent).
- The DoclingDocument JSON carries a `_enrichments.version` counter incremented by each stage. On load, a stage can verify it's reading the expected version.
- If a stage loads a JSON with an unexpected version (e.g., a prior stage was retried and re-wrote), it re-applies its enrichments from its own canonical source (DocumentElement rows).

This is NOT atomic. It is eventually consistent via idempotent retry. The same model already applies to the current pipeline (Postgres + MinIO writes are not atomic today).

### 2. Translation: original vs translated text (Findings 2, 3)

**Two representations preserved:**

- **Canonical `.text` on each item** — mutated to translated text so native tools (HybridChunker, Docling-Graph) see enriched content.
- **`_enrichments.translations[self_ref]`** — stores `{original_text, translated_text, language}` for each translated item.

**Markdown regeneration:**
- After translation, regenerate `docling_document.md` from the mutated DoclingDocument (translated content).
- Also regenerate `docling_document_original.md` from `_enrichments.translations` original texts. This preserves the classification-detection path that needs original markings.

**Metadata extraction** (`derive_document_metadata`):
- Uses translated markdown for summary/date/source (current behavior, unchanged).
- Uses original markdown for classification detection (reads `docling_document_original.md` if it exists, falls back to `docling_document.md`).

### 3. Document summary/classification context for graph extraction (Finding 2)

When loading the enriched DoclingDocument for graph extraction, the stage still prepends document summary and classification as a `_enrichments.context` field in the JSON. The Docling-Graph service can read this as supplementary context. This replaces the `_enriched_text` hack with a structured field.

### 4. Picture descriptions: annotation-only, no `.text` mutation (Finding 9)

Native `PictureItem` has no `.text` field. Descriptions persist ONLY as `annotations[{kind: "description"}]` (the `DescriptionAnnotation` type). This matches both the viewer's `<docling-tooltip>` contract and the native Docling schema. Native tools that process pictures will see the annotation; there is no `.text` to mutate.

### 5. HybridChunker instantiation (Finding 4)

```python
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

tok = AutoTokenizer.from_pretrained(settings.text_embedding_model)
hf_tokenizer = HuggingFaceTokenizer(tokenizer=tok, max_tokens=settings.chunk_max_tokens)
chunker = HybridChunker(tokenizer=hf_tokenizer)
```

Verified against installed `docling_core` API: `HybridChunker.tokenizer` expects `BaseTokenizer`, `HuggingFaceTokenizer` wraps `PreTrainedTokenizerBase` + `max_tokens`.

### 6. Translation toggle API contract (Finding 5)

`/element-translations` endpoint currently returns `{element_uid, original_text, translated_text}`. After refactor:
- Reads from `_enrichments.translations` (keyed by `self_ref`)
- Reverse-maps through `_enrichments.identity_map` to return `element_uid`
- Response shape unchanged — UI contract preserved

### 7. Graph extraction fallback (Finding 6)

If the DoclingDocument JSON is missing or corrupt:
- Fall back to reconstructing enriched text from DocumentElement rows (current behavior)
- Wrap in `{"_enriched_text": full_text}` as before
- Log a warning so the degradation is visible

### 8. DocumentElement schema: `self_ref` in `element_metadata` (Finding 7)

Store `self_ref` in the existing `element_metadata` JSONB column (no schema migration needed). Access via `element.element_metadata.get("self_ref")`. The identity_map in the JSON is the primary lookup; the JSONB field is a convenience for stages that already have a DocumentElement loaded.

### 9. Markdown regeneration schedule (Finding 8)

| Stage | Markdown artifacts updated |
|-------|---------------------------|
| `prepare_document` | `docling_document.md` (original), `docling_document.json` (original) |
| `detect_and_translate` | `docling_document.md` (now translated), `docling_document_original.md` (originals preserved), `docling_document.json` (v2: + translations) |
| `derive_picture_descriptions` | `docling_document.md` (regenerated with description appendix), `docling_document.json` (v3: + descriptions) |
| `derive_document_metadata` | No markdown changes (reads only) |
| `derive_text_chunks_and_embeddings` | No markdown changes (reads JSON) |
| `derive_ontology_graph` | No markdown changes (reads JSON) |

### 10. `/docling-raw` endpoint behavior change (Finding 6 from v1)

**Before:** Loads JSON, injects annotations at read time from Postgres.
**After:** Returns enriched JSON directly. Remove runtime injection. The annotation injection code in `sources.py` (lines ~817-836) is deleted since enrichments are now in the persisted JSON.

---

## Chunk schema mapping (Finding 5)

### Primary pass: HybridChunker → TextChunk rows

| Field | Source |
|-------|--------|
| `chunk_id` | Deterministic: `UUID(md5(sha256(doc_id:self_refs_joined:chunk_idx:model_version)))` |
| `document_id` | Pipeline context |
| `artifact_id` | Resolve: chunk's first `doc_item.self_ref` → `identity_map` → `element_uid` → Artifact row |
| `page_number` | `chunk.meta.doc_items[0].prov[0].page_no` if available, else None |
| `modality` | From source item label: `section_header`/`paragraph`/`list_item` → "text", `table` → "table" |
| `classification` | From document metadata |
| `chunk_text` | `chunk.text` (translated if translation ran) |
| `text_embedding` | BGE embedding of chunk_text |

### Secondary pass: Image description chunks

Scan `pictures` in DoclingDocument for items with `DescriptionAnnotation`. For each:
1. Extract description text from annotation
2. Split into sections via `split_description_sections()` (current logic)
3. Create TextChunk rows with `modality=image_description`
4. Create bidirectional `SAME_ARTIFACT` ChunkLinks between consecutive sections
5. Embed each section

### Fallback

If DoclingDocument reconstruction fails, fall back to `structure_aware_chunk()` over DocumentElement rows (current path). Log warning.

---

## Files changed

| File | Change |
|------|--------|
| `docker/docling/app/converter.py` | Extract `self_ref` from DocItems in `_extract_elements`, include in ConvertedElement metadata |
| `app/services/docling_client.py` | Pass `self_ref` through in ExtractedChunk metadata |
| `app/workers/pipeline.py` (prepare) | Build `_enrichments.identity_map`, persist in JSON. Store `self_ref` in DocumentElement.element_metadata |
| `app/workers/pipeline.py` (translate) | Reload JSON, mutate `.text` fields, persist `_enrichments.translations`, regenerate both markdown files, re-upload JSON |
| `app/workers/pipeline.py` (pictures) | Reload JSON, add `DescriptionAnnotation` to pictures, regenerate markdown, re-upload JSON |
| `app/workers/pipeline.py` (chunks) | Load JSON, reconstruct DoclingDocument, use HybridChunker with HuggingFaceTokenizer. Fallback to structure_aware_chunk |
| `app/workers/pipeline.py` (graph) | Load enriched JSON directly, add `_enrichments.context` with summary/classification, remove _enriched_text |
| `app/api/v1/sources.py` (docling-raw) | Remove runtime annotation injection |
| `app/api/v1/sources.py` (translations) | Read from `_enrichments.translations`, reverse-map via identity_map to element_uid |
| `app/services/chunking.py` | Keep as fallback; no changes |
