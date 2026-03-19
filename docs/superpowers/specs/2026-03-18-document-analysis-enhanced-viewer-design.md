# Document Analysis & Enhanced Viewer — Design Spec

## Goal

Add LLM-based document metadata extraction and picture description enrichment to the ingestion pipeline, switch Docling from VLM pipeline to PdfPipeline with dlparse_v4, and display extracted metadata above the document in the viewer.

## Architecture

The ingestion pipeline gains two new sequential Celery tasks between `prepare_document` and the existing parallel chord. The Docling service switches from `VlmPipeline` (granite-docling-258M) to `PdfPipelineOptions` with `dlparse_v4` backend, EasyOCR, TableFormer FAST, and formula/code enrichment. Picture descriptions move from the Docling service to a dedicated pipeline task that calls `gemma3:27b` via Ollama with the document summary as context.

## Pipeline Execution Order

```
prepare_document
  - Docling conversion (PdfPipeline + dlparse_v4, NO picture descriptions)
  - Persist elements, markdown, JSON to MinIO/Postgres

derive_document_metadata
  - Send markdown to gpt-oss:120b via Ollama
  - Extract: summary, date of information, classification, source characterization
  - Store in documents.document_metadata JSONB column

derive_picture_descriptions
  - Load PictureItem images from stored Docling JSON
  - For each image: call gemma3:27b via Ollama with summary-enriched prompt
  - Update Docling JSON in MinIO + DocumentElement rows with descriptions

chord(
  derive_text_chunks_and_embeddings,   # uses enriched picture descriptions
  derive_image_embeddings,
  derive_ontology_graph                # sees enriched descriptions + metadata
)

collect_derivations
derive_structure_links
derive_canonicalization
finalize_document
```

Text chunk embeddings use the updated picture descriptions. Graph extraction sees both the enriched picture descriptions and the document metadata (summary, classification), which can also be stored as properties in the graph.

## Docling Service Changes

Switch `docker/docling/app/converter.py` from `VlmPipeline` to standard `PdfPipeline`:

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableStructureOptions,
    TableFormerMode,
)

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=EasyOcrOptions(lang=OCR_LANGUAGES, use_gpu=True),
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.FAST,
    ),
    do_formula_enrichment=True,
    do_code_enrichment=True,
    generate_picture_images=True,
    generate_page_images=True,
    images_scale=1.0,
    do_picture_description=False,
)
```

- Granite VLM model no longer needed in Docling container
- EasyOCR language list configurable via `DOCLING_OCR_LANG` env var
- Fallback converter also uses PdfPipeline

## New Service: Document Analysis

New file `app/services/document_analysis.py` with two public functions:

### `extract_document_metadata(markdown: str) -> dict`

Calls Ollama `gpt-oss:120b` (configurable via `DOC_ANALYSIS_LLM_MODEL`) with four sequential prompts:

1. **Summary** — `DOC_ANALYSIS_SUMMARY_PROMPT` env var
2. **Date of information** — `DOC_ANALYSIS_DATE_PROMPT` env var
3. **Source characterization** — `DOC_ANALYSIS_SOURCE_PROMPT` env var
4. **Classification** — `DOC_ANALYSIS_CLASSIFICATION_PROMPT` env var

Returns:
```json
{
  "document_summary": "...",
  "date_of_information": "...",
  "classification": "UNCLASSIFIED",
  "source_characterization": "Organization: ...\nType: ...\nReliability: .../10",
  "generated_at": "2026-03-18T19:00:00+00:00"
}
```

### `describe_pictures(docling_json: dict, document_summary: str) -> dict`

Iterates picture items in the Docling JSON. For each picture with a base64 image:

1. Builds the prompt from `PICTURE_DESCRIPTION_PROMPT` env var, substituting `{document_summary}`
2. Sends image + prompt to Ollama `gemma3:27b` (configurable via `PICTURE_DESCRIPTION_MODEL`) at `{OLLAMA_BASE_URL}/v1/chat/completions`
3. Updates the picture item's description/caption in the Docling JSON

Returns the modified Docling JSON dict.

## New Pipeline Tasks

### `derive_document_metadata` (Celery task)

- Queue: `ingest`
- Input: `document_id`, `run_id`
- Loads markdown from MinIO (`artifacts/{document_id}/docling_document.md`)
- Calls `extract_document_metadata(markdown)`
- Stores result in `documents.document_metadata` JSONB column
- Chains to `derive_picture_descriptions`

### `derive_picture_descriptions` (Celery task)

- Queue: `ingest`
- Input: `document_id`, `run_id`
- Loads Docling JSON from MinIO (`artifacts/{document_id}/docling_document.json`)
- Loads `document_summary` from `documents.document_metadata`
- Calls `describe_pictures(docling_json, document_summary)`
- Writes updated Docling JSON back to MinIO
- Updates `DocumentElement` rows for picture elements with new descriptions
- Chains to the existing chord

## Database Migration

Alembic migration adding JSONB column:

```sql
ALTER TABLE ingest.documents ADD COLUMN document_metadata JSONB DEFAULT NULL;
```

## Environment Variables

All added to `.env` and `.env.example`:

```env
# Document Analysis (LLM metadata extraction after Docling conversion)
DOC_ANALYSIS_ENABLED=true
DOC_ANALYSIS_LLM_MODEL=gpt-oss:120b
DOC_ANALYSIS_TIMEOUT=300

DOC_ANALYSIS_SUMMARY_PROMPT=Summarize this document in 3-5 sentences for a technical reader. Focus on the main subject, scope, and notable findings. Do not include source endnote markings such as [1].
DOC_ANALYSIS_DATE_PROMPT=Extract the most relevant date of information (publication date, report date, or coverage window). If only month/year appears, return that. If there is a range, return it exactly. If unsure, return Unknown. Provide ONLY the date or range.
DOC_ANALYSIS_SOURCE_PROMPT=Characterize the source: 1) Organization or author (if unknown return UNKNOWN) 2) Type of information (website, journal, etc; if unknown return UNKNOWN) 3) Reliability score 1-10. Format: Organization: <name>\nType: <type>\nReliability: <score>/10
DOC_ANALYSIS_CLASSIFICATION_PROMPT=Identify the document classification marking if present (UNCLASSIFIED, CUI, FOUO, SECRET, TOP SECRET). If none, reply UNCLASSIFIED. Provide ONLY the marking.

# Picture Description (post-conversion via Ollama, uses document summary as context)
PICTURE_DESCRIPTION_MODEL=gemma3:27b
PICTURE_DESCRIPTION_TIMEOUT=120
PICTURE_DESCRIPTION_PROMPT=Analyze this image from a multi-modal PDF using the required narrative sections and the missile/radar/S&T emphasis. Return sections 1-8 exactly as specified. Use the PDF Summary for context but rely on visual evidence.\n\n- PDF Summary: {document_summary}\n\n- Image:

# Docling OCR
DOCLING_OCR_LANG=en
```

Corresponding `app/config.py` fields with defaults matching these values.

## Frontend: Metadata Panel in DoclingViewer

### API Endpoint

New `GET /v1/documents/{document_id}/metadata` returns the `document_metadata` JSONB or 404 if not yet extracted.

### DoclingViewer Changes

`frontend/src/components/DoclingViewer.tsx`:

- Fetch metadata on mount alongside the Docling JSON
- Render metadata panel **above** the document viewer (inside the modal body, before the iframe)
- Only shown when metadata exists

Layout:
```
┌─────────────────────────────────────────────────┐
│ [filename]                    [Document] [JSON] X│
├─────────────────────────────────────────────────┤
│ ┌─ AI-Extracted Document Metadata ────────────┐ │
│ │ Classification: UNCLASSIFIED                │ │
│ │ Date of Information: April 2020             │ │
│ │ Source: Organization: ... | Type: ... | 8/10│ │
│ │ Summary: This document covers the S-75...   │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Document Viewer (docling-img) ─────────────┐ │
│ │  [page images with bounding boxes]          │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

Styled with existing design system: card with `var(--color-border)` border, muted labels, section divider.

### API Client

New function in `frontend/src/api/client.ts`:
```typescript
export async function getDocumentMetadata(documentId: string): Promise<Record<string, unknown> | null>
```

## Files Changed/Created

| File | Action | Purpose |
|------|--------|---------|
| `app/services/document_analysis.py` | Create | LLM metadata extraction + picture description logic |
| `app/workers/pipeline.py` | Modify | Add two new tasks, rechain pipeline |
| `app/api/v1/sources.py` | Modify | Add metadata endpoint |
| `app/config.py` | Modify | Add new settings fields |
| `app/models/ingest.py` | Modify | Add `document_metadata` column to Document |
| `docker/docling/app/converter.py` | Modify | Switch to PdfPipeline + dlparse_v4 |
| `frontend/src/components/DoclingViewer.tsx` | Modify | Add metadata panel |
| `frontend/src/api/client.ts` | Modify | Add metadata fetch |
| `.env` | Modify | Add new env vars |
| `alembic/versions/XXXX_add_document_metadata.py` | Create | Migration |
