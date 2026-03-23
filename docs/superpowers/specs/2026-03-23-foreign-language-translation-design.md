# Foreign Language Detection and Translation — Design Spec

**Date:** 2026-03-23
**Status:** Approved

## Overview

Detect non-English content in document text and OCR output during ingest. When detected, translate paragraph-by-paragraph via LLM, preserving markdown structure. Store both original and translated versions. All downstream pipeline stages operate on the English translation. The DoclingViewer shows the original by default with a "Translate" toggle that switches to a rendered markdown view of the English translation.

Image descriptions are excluded — the picture description prompt already handles foreign-language text transcription and translation.

## Pipeline Position

```
prepare_document (Docling extracts text/OCR)
  ↓
detect_and_translate (NEW — language detect + LLM translate if non-English)
  ↓
derive_document_metadata (uses translated markdown when available)
  ↓
derive_picture_descriptions (unchanged — prompt handles foreign text)
  ↓
[rest of pipeline uses translated text]
```

New Celery task `detect_and_translate` on the `ingest` queue, positioned after `prepare_document` and before `derive_document_metadata` in the existing chain.

## Language Detection

**Library:** `langdetect` (lightweight, no model download required).

Detection runs on the extracted markdown from MinIO (`docling_document.md`). OCR text is already embedded inline in the markdown by Docling, so it is covered by the same detection pass.

**Logic:**
1. Load `docling_document.md` from MinIO
2. Sample the first ~5000 characters (sufficient for reliable detection, fast)
3. Run `langdetect.detect_langs()` to get language probabilities
4. If top language is not `en` with confidence > 0.7, mark the document for translation
5. Store detected language code in `documents.document_metadata` JSONB as `"detected_language": "ru"` (ISO 639-1)
6. If English detected or confidence is low, skip translation — pipeline continues with zero added latency

## Translation

When non-English content is detected:

### Paragraph-by-Paragraph Translation

1. Split markdown on `\n\n` (paragraph boundaries)
2. Skip purely structural lines (e.g., `---`, empty lines, image references `![...]`)
3. Batch consecutive paragraphs up to ~2000 characters per LLM call to reduce call count
4. Each batch is sent to the LLM with a system prompt instructing it to:
   - Translate to English
   - Preserve markdown formatting (headings, bullets, table structure)
   - Preserve technical designators verbatim (e.g., С-75, ЗРК, 9М38)
   - Preserve numbers, units, and acronyms
5. Reassemble translated paragraphs into `docling_document_translated.md`, maintaining original structure

### Configuration

| Env Var | Default | Description |
|---|---|---|
| `TRANSLATION_ENABLED` | `true` | Enable/disable translation pipeline stage |
| `TRANSLATION_MODEL` | `gpt-oss:120b` | LLM model for translation |
| `TRANSLATION_TIMEOUT` | `300` | Per-call timeout (seconds) |
| `TRANSLATION_PROMPT` | See below | System prompt for translation |
| `TRANSLATION_SOFT_TIME_LIMIT` | `3600` | Celery soft time limit (seconds) |
| `TRANSLATION_TIME_LIMIT` | `3660` | Celery hard time limit (seconds) |

**Default translation prompt:**
```
Translate the following text to English. Preserve all markdown formatting including headings, bullet points, tables, and code blocks. Preserve technical designators, model numbers, NATO reporting names, and military identifiers verbatim — do not transliterate or translate them. Preserve all numbers, units, and acronyms. Return only the translated text with no commentary.
```

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

## Storage

| Artifact | Path | When |
|---|---|---|
| Original markdown | `artifacts/{doc_id}/docling_document.md` | Always (unchanged) |
| Original JSON | `artifacts/{doc_id}/docling_document.json` | Always (unchanged) |
| Translated markdown | `artifacts/{doc_id}/docling_document_translated.md` | Only when non-English detected |

**Document metadata** gains two keys in the existing `document_metadata` JSONB column:

- `detected_language`: ISO 639-1 code (e.g., `"ru"`, `"zh-cn"`, `"en"`)
- `has_translation`: boolean (`true` if translated markdown exists)

No schema migration needed — these are JSONB keys added to the existing nullable column.

## Downstream Pipeline Impact

Two tasks need to prefer translated markdown when available:

### `derive_document_metadata`

When loading markdown from MinIO, try `docling_document_translated.md` first, fall back to `docling_document.md`:

```python
base_key = f"artifacts/{document_id}"
try:
    md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document_translated.md")
except Exception:
    md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.md")
markdown = md_bytes.decode("utf-8")
```

### `derive_text_chunks_and_embeddings`

Same pattern — load translated markdown for the element text that feeds into chunking and BGE embedding. The existing code reads from `DocumentElement.content_text` (set during `prepare_document` from the original text). Two approaches:

**Approach A:** Update `DocumentElement.content_text` with translated text during `detect_and_translate`. This means downstream tasks work unchanged.

**Approach B:** Have `derive_text_chunks_and_embeddings` load the translated markdown and use it instead of element content_text.

**Recommendation: Approach A.** Update the DocumentElement rows with translated text during the translation stage. Store the original text in a new JSONB field `original_content` on DocumentElement (only populated when translation occurs). This keeps all downstream code unchanged and preserves the original.

Wait — this requires a schema migration for the `original_content` field. Simpler alternative: store the original-to-translated mapping only in MinIO (the two markdown files). For `derive_text_chunks_and_embeddings`, re-read the translated markdown and re-map it to elements by matching structure. This is fragile.

**Revised recommendation: Approach A with no new column.** The original text is already preserved in `docling_document.md` in MinIO. During `detect_and_translate`, update `DocumentElement.content_text` with the translated text for non-English elements. The original is recoverable from MinIO. Downstream tasks operate on the translated content_text unchanged.

## API

### Existing Endpoint

`GET /v1/documents/{documentId}/metadata` — already returns the full `document_metadata` JSONB. Clients can read `detected_language` and `has_translation` from it. No changes needed.

### New Endpoint

`GET /v1/documents/{documentId}/translation`

Returns the translated markdown as plain text. Response:

```json
{
  "document_id": "...",
  "detected_language": "ru",
  "translated_markdown": "# Translated content..."
}
```

Returns 404 if no translation exists for this document.

**File:** `app/api/v1/sources.py`

## DoclingViewer Changes

**File:** `frontend/src/components/DoclingViewer.tsx`

### Toggle Button

When `metadata.has_translation` is `true`, add a "Translate" button to the mode toggle bar (alongside "Document" and "JSON"):

```
[Document] [Translate] [JSON]
```

### Translation View

When "Translate" mode is active:
1. Fetch translated markdown from `GET /v1/documents/{documentId}/translation` (cached in component state after first fetch)
2. Render as styled markdown in a `<pre style="white-space: pre-wrap">` block (same pattern as the existing plain-text fallback at DoclingViewer.tsx line 211)
3. Show a banner at the top: "Machine-translated from {language_name}. Original may contain untranslated technical terms."

### Default State

Original Docling page-layout rendering is the default. Translation is opt-in via toggle.

## What Doesn't Change

- Image descriptions — prompt already handles foreign-language text
- Docling JSON structure — original preserved as-is in MinIO
- CLIP image embeddings — pixel-based, language-independent
- GraphRAG indexing — operates on text chunks which are already English after translation
- Existing English documents — detection runs, finds English, skips translation with zero added latency
- Pipeline chain structure — new task is inserted into the existing chain, no new parallel paths

## Error Handling

- If translation fails mid-document (LLM timeout, Ollama unavailable), the task retries once. On final failure, the document continues through the pipeline with original text — metadata and embeddings will be in the original language, which is degraded but not broken (BGE-M3 is multilingual).
- `has_translation` is only set to `true` after the translated markdown is successfully written to MinIO.
- Partial translations are not stored — it's all or nothing for a given document.
