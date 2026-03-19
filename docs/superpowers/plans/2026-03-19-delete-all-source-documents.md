# Delete All Source Documents Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the UI "Clear list" button with a "Delete All" button that deletes all documents in the current source and all associated data (Postgres, Qdrant, Neo4j, MinIO).

**Architecture:** New `DELETE /v1/sources/{source_id}/documents` endpoint iterates all source documents and calls the existing `_hard_delete_document()` for each. Frontend calls this endpoint with a confirmation dialog.

**Tech Stack:** FastAPI (backend), React/TypeScript (frontend), existing `_hard_delete_document` cleanup function.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `app/api/v1/sources.py` | Modify | Add `DELETE /sources/{source_id}/documents` endpoint |
| `frontend/src/api/client.ts` | Modify | Add `deleteAllSourceDocuments()` function |
| `frontend/src/components/FileUpload.tsx` | Modify | Replace "Clear list" with "Delete All" button + confirm |

---

## Chunk 1: Backend + Frontend

### Task 1: Add DELETE /sources/{source_id}/documents backend endpoint

**Files:**
- Modify: `app/api/v1/sources.py` (after `delete_document` endpoint, ~line 419)

- [ ] **Step 1: Add the endpoint**

Add after the existing `delete_document` endpoint (~line 419):

```python
@router.delete("/sources/{source_id}/documents")
async def delete_all_source_documents(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Delete all documents in a source and all their derived data."""
    source = await db.get(Source, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")

    docs = (await db.execute(
        select(Document).where(Document.source_id == source_id)
    )).scalars().all()

    if not docs:
        return {"deleted": 0}

    processing = [d for d in docs if d.pipeline_status == "PROCESSING"]
    if processing:
        raise HTTPException(
            status_code=409,
            detail=f"{len(processing)} document(s) still processing. Cancel them first.",
        )

    count = 0
    for doc in docs:
        try:
            await _hard_delete_document(doc.id, doc, db)
            count += 1
        except Exception as exc:
            logger.warning("delete_all: failed to delete %s: %s", doc.id, exc)

    logger.info("delete_all_source_documents: deleted %d/%d documents from source %s", count, len(docs), source_id)
    return {"deleted": count}
```

- [ ] **Step 2: Verify the server starts**

Run: `docker compose up -d api && docker compose logs api --tail 20`
Expected: No import errors, server starts successfully.

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/sources.py
git commit -m "feat: add DELETE /sources/{source_id}/documents endpoint for batch delete"
```

---

### Task 2: Add deleteAllSourceDocuments to frontend API client

**Files:**
- Modify: `frontend/src/api/client.ts` (after `deleteDocument` function, ~line 409)

- [ ] **Step 1: Add the client function**

Add after the existing `deleteDocument` function:

```typescript
export async function deleteAllSourceDocuments(sourceId: string): Promise<{ deleted: number }> {
  const res = await fetch(`/v1/sources/${sourceId}/documents`, { method: "DELETE" });
  return handleResponse<{ deleted: number }>(res);
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat: add deleteAllSourceDocuments API client function"
```

---

### Task 3: Replace "Clear list" with "Delete All" button in FileUpload

**Files:**
- Modify: `frontend/src/components/FileUpload.tsx`
  - Line 3: Add `deleteAllSourceDocuments` to imports
  - Lines 500-509: Replace the "Clear list" button

- [ ] **Step 1: Add import**

In the import block at line 3, add `deleteAllSourceDocuments`:

```typescript
import {
  batchDocumentStatus,
  cancelDocument,
  createSource,
  deleteAllSourceDocuments,
  deleteDocument,
  listDocumentsBySource,
  listSources,
  reingestDocument,
  uploadFile,
  type Document,
  type Source,
} from "../api/client";
```

- [ ] **Step 2: Replace the "Clear list" button**

Replace the block at lines 500-509:

```tsx
{entries.length > 0 && (
  <div style={{ marginTop: "1rem", textAlign: "right" }}>
    <button
      className="btn btn-ghost btn-sm"
      onClick={() => setEntries([])}
    >
      Clear list
    </button>
  </div>
)}
```

With:

```tsx
{selectedSourceId && (existingDocs.length > 0 || entries.some((e) => e.documentId)) && (
  <div style={{ marginTop: "1rem", textAlign: "right" }}>
    <button
      className="btn btn-danger btn-sm"
      onClick={async () => {
        const sourceName = sources.find((s) => s.id === selectedSourceId)?.name ?? "this source";
        const totalDocs = existingDocs.length + entries.filter((e) => e.documentId).length;
        if (!confirm(
          `Delete all ${totalDocs} document(s) in "${sourceName}"? ` +
          "This removes all files, extractions, embeddings, and graph data. This cannot be undone."
        )) return;
        try {
          await deleteAllSourceDocuments(selectedSourceId);
          setEntries([]);
          setExistingDocs([]);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Delete all failed");
        }
      }}
    >
      Delete All
    </button>
  </div>
)}
```

- [ ] **Step 3: Verify in browser**

1. Navigate to the ingest UI
2. Select a source with documents
3. Confirm "Delete All" button appears with danger styling
4. Click it — confirm dialog should show document count and source name
5. Confirm — documents should be removed from all views

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/FileUpload.tsx
git commit -m "feat: replace Clear list with Delete All button that deletes all source documents"
```
