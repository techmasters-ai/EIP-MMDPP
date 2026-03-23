# Foreign Language Detection and Translation — Design Spec

**Date:** 2026-03-23
**Status:** Approved

## Overview

Detect non-English content in document text and OCR output during ingest. When detected, translate element-by-element via LLM, preserving markdown structure. Store both original and translated versions. All downstream pipeline stages operate on the English translation. The DoclingViewer shows the original by default with a "Translate" toggle that switches to a rendered markdown view of the English translation.

Image descriptions are excluded — the picture description prompt already handles foreign-language text transcription and translation.

## Pipeline Position

The new `detect_and_translate` task is inserted into the Celery chain between `prepare_document` and the first chord:

```python
# Current chain (simplified):
chain(
    prepare_document,
    chord(group(derive_document_metadata, purge_document_derivations), derive_picture_descriptions),
    chord(group(text_embed, image_embed, ontology_graph), collect_derivations),
    derive_structure_links,
    derive_canonicalization,
    finalize_document,
)

# Updated chain:
chain(
    prepare_document,
    detect_and_translate,  # NEW
    chord(group(derive_document_metadata, purge_document_derivations), derive_picture_descriptions),
    chord(group(text_embed, image_embed, ontology_graph), collect_derivations),
    derive_structure_links,
    derive_canonicalization,
    finalize_document,
)
```

New Celery task `detect_and_translate` on the `ingest` queue.

## Language Detection

**Library:** `langdetect` (lightweight, no model download required). Must be seeded for deterministic results: `DetectorFactory.seed = 0`.

Detection runs on the extracted markdown from MinIO (`docling_document.md`). OCR text is already embedded inline in the markdown by Docling, so it is covered by the same detection pass.

**Logic:**
1. Load `docling_document.md` from MinIO
2. If text is shorter than 50 characters, skip detection (too short for reliable results) — set `detected_language: "unknown"` and continue
3. Sample the first ~5000 characters
4. Run `langdetect.detect_langs()` to get language probabilities
5. If top language is not `en` with confidence > 0.7, mark the document for translation
6. Store detected language code in `documents.document_metadata` JSONB as `"detected_language": "ru"` (ISO 639-1)
7. If English detected or confidence is low, skip translation — pipeline continues with zero added latency

**Mixed-language documents:** The spec uses document-level detection (first 5000 chars). If a document is half Russian and half English, the dominant language in the sample determines the action. When translation is triggered, **all** non-structural content is sent to the LLM — the translation prompt instructs the LLM to translate non-English text and pass through English text unchanged. This handles mixed-language documents naturally without paragraph-level detection.

## Translation

When non-English content is detected:

### Element-by-Element Translation

Translation operates directly on `DocumentElement` rows rather than the flat markdown file. This avoids the fragile problem of mapping translated paragraphs back to element rows.

1. Query all `DocumentElement` rows for the document where `content_text IS NOT NULL` and `element_type IN ('text', 'heading', 'table', 'equation')`
2. Batch elements by concatenating their `content_text` with `\n---ELEMENT_BOUNDARY---\n` separators, up to ~2000 characters per batch
3. Send each batch to the LLM with the translation prompt, instructing it to preserve the `---ELEMENT_BOUNDARY---` markers
4. Parse the response back into individual translated texts by splitting on the boundary marker
5. Update each `DocumentElement.content_text` with the translated text
6. Reassemble the translated elements into `docling_document_translated.md` (for the viewer endpoint) by joining them in `element_order`
7. Upload `docling_document_translated.md` to MinIO

**The original text is preserved in `docling_document.md` in MinIO** (never modified by this task). The `DocumentElement.content_text` is updated with translated text so all downstream tasks (metadata, chunking, embeddings, ontology graph) operate on English without code changes.

### Configuration

All settings added to the `Settings` class in `app/config.py`:

```python
# Translation (foreign language detection + LLM translation)
translation_enabled: bool = True
translation_model: str = "gpt-oss:120b"
translation_timeout: int = 300
translation_prompt: str = "Translate the following text to English. ..."
translation_soft_time_limit: int = 3600
translation_time_limit: int = 3660
```

| Env Var | Config Field | Default | Description |
|---|---|---|---|
| `TRANSLATION_ENABLED` | `translation_enabled` | `true` | Enable/disable translation stage |
| `TRANSLATION_MODEL` | `translation_model` | `gpt-oss:120b` | LLM model for translation |
| `TRANSLATION_TIMEOUT` | `translation_timeout` | `300` | Per-call timeout (seconds) |
| `TRANSLATION_PROMPT` | `translation_prompt` | See below | System prompt for translation |
| `TRANSLATION_SOFT_TIME_LIMIT` | `translation_soft_time_limit` | `3600` | Celery soft time limit |
| `TRANSLATION_TIME_LIMIT` | `translation_time_limit` | `3660` | Celery hard time limit |

**Default translation prompt:**
```
Translate the following text to English. If the text is already in English, return it unchanged. Preserve all markdown formatting including headings (#), bullet points, tables, and code blocks. Preserve technical designators, model numbers, NATO reporting names, and military identifiers verbatim — do not transliterate or translate them (e.g., keep С-75, ЗРК, 9М38 as-is). Preserve all numbers, units, and acronyms. Preserve ---ELEMENT_BOUNDARY--- markers exactly as they appear. Return only the translated text with no commentary.
```

### Oversized Elements

If a single element's `content_text` exceeds the ~2000 character batch limit (e.g., a large table), it is sent as its own batch. The LLM context window can handle individual elements up to `ollama_num_ctx * 3` characters (same truncation logic used by `derive_document_metadata`).

### Task Definition

```python
@celery_app.task(
    bind=True,
    name="app.workers.pipeline.detect_and_translate",
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=settings.translation_soft_time_limit,
    time_limit=settings.translation_time_limit,
    queue="ingest",
)
```

The task must also be added to `REQUIRED_STAGES` in `finalize_document` so translation failures are reflected in the pipeline status.

### Stage Run Metrics

The translation stage records metrics in `StageRun.metrics`:
```json
{
  "detected_language": "ru",
  "confidence": 0.95,
  "elements_translated": 42,
  "batch_count": 8,
  "skipped": false
}
```

## Storage

| Artifact | Path | When |
|---|---|---|
| Original markdown | `artifacts/{doc_id}/docling_document.md` | Always (unchanged, never modified by translation) |
| Original JSON | `artifacts/{doc_id}/docling_document.json` | Always (unchanged) |
| Translated markdown | `artifacts/{doc_id}/docling_document_translated.md` | Only when non-English detected and translation succeeds |

**Document metadata** gains two keys in the existing `document_metadata` JSONB column:

- `detected_language`: ISO 639-1 code (e.g., `"ru"`, `"zh-cn"`, `"en"`)
- `has_translation`: boolean (`true` if translated markdown exists)

No schema migration needed — these are JSONB keys added to the existing nullable column.

## Downstream Pipeline Impact

Because `detect_and_translate` updates `DocumentElement.content_text` directly, **all downstream tasks operate on translated text without code changes:**

| Task | How It Reads Text | Impact |
|---|---|---|
| `derive_document_metadata` | Loads `docling_document.md` from MinIO | **Needs change:** try `docling_document_translated.md` first, fall back to original |
| `derive_picture_descriptions` | Reads `DocumentElement.content_text` for image elements | No change needed — image elements are excluded from translation |
| `derive_text_chunks_and_embeddings` | Reads `DocumentElement.content_text` | No change needed — already gets translated text |
| `derive_ontology_graph` | Reads `DocumentElement.content_text` and concatenates | No change needed — already gets translated text |
| `derive_image_embeddings` | Reads image pixel data | No change needed — pixel-based |
| `derive_picture_descriptions` → markdown enrichment | Appends to `docling_document.md` | Appends to the **original** markdown only. The translated markdown does not get image description appendix — this is acceptable because image descriptions are already in English |

### `derive_document_metadata` — special handling

Metadata extraction should use the **translated markdown** for the summary prompt (so the summary is in English) but the **original markdown** for classification marking detection (markings like СЕКРЕТНО should be detected in their original form). Implementation:

```python
# Load translated for summary, original for classification
try:
    translated_md = download_bytes_sync(bucket, f"{base_key}/docling_document_translated.md").decode()
except Exception:
    translated_md = None
original_md = download_bytes_sync(bucket, f"{base_key}/docling_document.md").decode()

# Use translated for summary/date/source prompts; original for classification
summary_text = translated_md or original_md
classification_text = original_md
```

## Cleanup

### Document Deletion

`_hard_delete_document` in `app/api/v1/sources.py` must also delete `docling_document_translated.md` from MinIO alongside the existing `docling_document.md` and `docling_document.json` cleanup.

### `purge_document_derivations`

When a document is re-ingested, `purge_document_derivations` should delete the translated markdown from MinIO to avoid stale translations.

## API

### Existing Endpoint

`GET /v1/documents/{document_id}/metadata` — already returns the full `document_metadata` JSONB. Clients can read `detected_language` and `has_translation` from it. No changes needed.

### New Endpoint

`GET /v1/documents/{document_id}/translation`

**File:** `app/api/v1/sources.py`

Returns the translated markdown. Response:

```json
{
  "document_id": "...",
  "detected_language": "ru",
  "translated_markdown": "# Translated content..."
}
```

Returns 404 if no translation exists for this document.

## DoclingViewer Changes

**File:** `frontend/src/components/DoclingViewer.tsx`

### Toggle Button

When `metadata.has_translation` is `true`, add a "Translate" button to the mode toggle bar (alongside "Document" and "JSON"):

```
[Document] [Translate] [JSON]
```

### Translation View

When "Translate" mode is active:
1. Fetch translated markdown from `GET /v1/documents/{document_id}/translation` (cached in component state after first fetch)
2. Render as formatted HTML using a lightweight markdown renderer (e.g., `react-markdown` or `marked`) for proper heading/table/list rendering
3. Show a banner at the top: "Machine-translated from {language_name}. Original may contain untranslated technical terms."

### Default State

Original Docling page-layout rendering is the default. Translation is opt-in via toggle.

## What Doesn't Change

- Image descriptions — prompt already handles foreign-language text
- Docling JSON structure — original preserved as-is in MinIO
- CLIP image embeddings — pixel-based, language-independent
- GraphRAG indexing — operates on text chunks which are already English after translation
- Existing English documents — detection runs, finds English, skips translation with zero added latency

## Error Handling

- If translation fails mid-document (LLM timeout, Ollama unavailable), the task retries once. On final failure, the document continues through the pipeline with original (untranslated) text in `DocumentElement.content_text`. Metadata and embeddings will be in the original language — retrieval quality for that document will be **severely degraded** since the embedding model (`BAAI/bge-large-en-v1.5`) is English-only. The document will still be indexed but English queries will match poorly.
- `has_translation` is only set to `true` after the translated markdown is successfully written to MinIO and `DocumentElement` rows are updated.
- Partial translations are not stored — it's all or nothing for a given document.
- If the boundary marker parsing fails (LLM strips or mangles markers), fall back to translating each element individually (slower but reliable).
