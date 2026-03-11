import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  batchDocumentStatus,
  createSource,
  listDocumentsBySource,
  listSources,
  reingestDocument,
  uploadFile,
  type Document,
  type Source,
} from "../api/client";

export interface FileEntry {
  file: File;
  fileName: string;        // cached for display after File ref may be gone
  fileSize: number;        // cached for display after File ref may be gone
  progress: number;        // 0–100 upload progress
  status: "queued" | "uploading" | "polling" | "COMPLETE" | "ERROR" | string;
  documentId?: string;
  error?: string;
}

interface FileUploadProps {
  entries: FileEntry[];
  setEntries: React.Dispatch<React.SetStateAction<FileEntry[]>>;
  selectedSourceId: string;
  setSelectedSourceId: React.Dispatch<React.SetStateAction<string>>;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "COMPLETE"
      ? "badge badge-complete"
      : status === "ERROR" || status === "FAILED"
      ? "badge badge-error"
      : status === "PARTIAL_COMPLETE" || status === "PENDING_HUMAN_REVIEW"
      ? "badge badge-warning"
      : status === "uploading"
      ? "badge badge-processing"
      : status === "polling" || status === "PROCESSING"
      ? "badge badge-processing"
      : "badge badge-pending";

  const label =
    status === "queued"
      ? "Queued"
      : status === "uploading"
      ? "Uploading"
      : status === "polling"
      ? "Processing…"
      : status === "PARTIAL_COMPLETE"
      ? "Partial"
      : status === "FAILED"
      ? "Failed"
      : status === "PENDING_HUMAN_REVIEW"
      ? "Needs Review"
      : status;

  return <span className={cls}>{label}</span>;
}

export function FileUpload({ entries, setEntries, selectedSourceId, setSelectedSourceId }: FileUploadProps) {
  const [sources, setSources] = useState<Source[]>([]);
  const [newSourceName, setNewSourceName] = useState("");
  const [directoryMode, setDirectoryMode] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [existingDocs, setExistingDocs] = useState<Document[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load sources on mount — don't clobber persisted selectedSourceId
  useEffect(() => {
    listSources()
      .then((s) => {
        setSources(s);
        if (!selectedSourceId && s.length > 0) setSelectedSourceId(s[0].id);
      })
      .catch(() => {/* sources list optional */});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load existing documents when selected source changes
  useEffect(() => {
    if (!selectedSourceId) {
      setExistingDocs([]);
      return;
    }
    listDocumentsBySource(selectedSourceId)
      .then(setExistingDocs)
      .catch(() => setExistingDocs([]));
  }, [selectedSourceId]);

  // Poll pipeline status using batch endpoint with adaptive interval.
  // Dependency is the set of pending document IDs (not entries itself),
  // so status updates from polling don't reset the timer/startTime.
  const TERMINAL = new Set(["COMPLETE", "ERROR", "FAILED", "PARTIAL_COMPLETE", "PENDING_HUMAN_REVIEW"]);
  const pendingEntryIds = entries
    .filter((e) => e.documentId && !TERMINAL.has(e.status))
    .map((e) => e.documentId!);
  const pendingExistingIds = existingDocs
    .filter((d) => !TERMINAL.has(d.pipeline_status))
    .map((d) => d.id);
  const pendingIdsKey = [...pendingEntryIds, ...pendingExistingIds]
    .sort()
    .join(",");
  const pollStartRef = useRef<number>(0);

  useEffect(() => {
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    if (!pendingIdsKey) return;

    // Reset start time only when the set of pending IDs changes
    pollStartRef.current = Date.now();
    const pendingIds = pendingIdsKey.split(",");

    const getInterval = () => {
      const elapsed = Date.now() - pollStartRef.current;
      if (elapsed < 30_000) return 2000;
      if (elapsed < 120_000) return 5000;
      return 10_000;
    };

    const poll = async () => {
      try {
        const docs = await batchDocumentStatus(pendingIds);
        const statusMap = new Map(docs.map((d) => [d.id, d]));
        setEntries((prev) =>
          prev.map((entry) => {
            if (!entry.documentId) return entry;
            const doc = statusMap.get(entry.documentId);
            if (!doc) return entry;
            return { ...entry, status: doc.pipeline_status, error: doc.error_message || entry.error };
          }),
        );
        // Also update existing docs with new statuses
        setExistingDocs((prev) =>
          prev.map((d) => {
            const updated = statusMap.get(d.id);
            return updated ? { ...d, pipeline_status: updated.pipeline_status, error_message: updated.error_message } : d;
          }),
        );
      } catch {
        // Batch endpoint failed — skip this poll cycle
      }
    };

    const scheduleNext = () => {
      pollTimerRef.current = setTimeout(async () => {
        await poll();
        scheduleNext();
      }, getInterval());
    };
    scheduleNext();

    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, [pendingIdsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const ensureSource = useCallback(async (): Promise<string | null> => {
    if (selectedSourceId) return selectedSourceId;
    const name = newSourceName.trim();
    if (!name) {
      setError("Select or create a source before uploading.");
      return null;
    }
    try {
      const src = await createSource(name);
      setSources((prev) => [...prev, src]);
      setSelectedSourceId(src.id);
      setNewSourceName("");
      return src.id;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create source");
      return null;
    }
  }, [selectedSourceId, newSourceName]);

  const processFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setError(null);

      const sourceId = await ensureSource();
      if (!sourceId) return;

      const newEntries: FileEntry[] = files.map((f) => ({
        file: f,
        fileName: f.name,
        fileSize: f.size,
        progress: 0,
        status: "queued",
      }));
      setEntries((prev) => [...prev, ...newEntries]);

      // Upload sequentially to avoid hammering the server
      for (let i = 0; i < newEntries.length; i++) {
        const idx = entries.length + i;
        setEntries((prev) =>
          prev.map((e, j) => (j === idx ? { ...e, status: "uploading" } : e)),
        );

        try {
          const doc = await uploadFile(sourceId, newEntries[i].file, (pct) => {
            setEntries((prev) =>
              prev.map((e, j) => (j === idx ? { ...e, progress: pct } : e)),
            );
          });
          setEntries((prev) =>
            prev.map((e, j) =>
              j === idx
                ? { ...e, progress: 100, status: "polling", documentId: doc.id }
                : e,
            ),
          );
        } catch (err) {
          setEntries((prev) =>
            prev.map((e, j) =>
              j === idx
                ? {
                    ...e,
                    status: "ERROR",
                    error: err instanceof Error ? err.message : "Upload failed",
                  }
                : e,
            ),
          );
        }
      }
    },
    [entries.length, ensureSource],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      void processFiles(files);
    },
    [processFiles],
  );

  const onFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      void processFiles(files);
      // Reset input so same files can be re-selected
      e.target.value = "";
    },
    [processFiles],
  );

  return (
    <div>
      {error && <div className="alert alert-error">{error}</div>}

      {/* Source selection */}
      <div className="card card-body" style={{ marginBottom: "1rem" }}>
        <div className="field-row">
          <div className="field">
            <label htmlFor="source-select">Source</label>
            <select
              id="source-select"
              value={selectedSourceId}
              onChange={(e) => setSelectedSourceId(e.target.value)}
            >
              <option value="">— new source —</option>
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          {!selectedSourceId && (
            <div className="field">
              <label htmlFor="new-source">New source name</label>
              <input
                id="new-source"
                type="text"
                placeholder="Program / collection name"
                value={newSourceName}
                onChange={(e) => setNewSourceName(e.target.value)}
              />
            </div>
          )}
        </div>

        <div className="flex-center gap-sm mt-sm">
          <input
            type="checkbox"
            id="dir-mode"
            checked={directoryMode}
            onChange={(e) => setDirectoryMode(e.target.checked)}
          />
          <label htmlFor="dir-mode" style={{ margin: 0 }}>
            Upload entire directory
          </label>
        </div>
      </div>

      {/* Drop zone */}
      <div
        className={`drop-zone${dragOver ? " drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="drop-zone-icon">📄</div>
        <div className="drop-zone-label">
          {directoryMode
            ? "Click to select a directory, or drag & drop files here"
            : "Click to select files, or drag & drop here"}
        </div>
        <div className="drop-zone-sub">PDF, DOCX, TXT, PNG, JPG, TIFF supported</div>

        <input
          ref={fileInputRef}
          type="file"
          multiple={!directoryMode}
          // @ts-expect-error — webkitdirectory is non-standard but widely supported
          webkitdirectory={directoryMode ? "" : undefined}
          style={{ display: "none" }}
          onChange={onFileInput}
          accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.png,.jpg,.jpeg,.tiff,.tif"
        />
      </div>

      {/* File list */}
      {entries.length > 0 && (
        <div className="file-list">
          {entries.map((entry, i) => (
            <div key={i} className="file-item">
              <span className="file-item-name" title={entry.fileName}>
                {entry.fileName}
              </span>
              <span className="file-item-size">{formatBytes(entry.fileSize)}</span>

              {entry.status === "uploading" ? (
                <div className="progress-wrap">
                  <div className="progress-bar" style={{ width: `${entry.progress}%` }} />
                </div>
              ) : null}

              <StatusBadge status={entry.status} />

              {(entry.status === "FAILED" || entry.status === "ERROR" || entry.status === "PENDING") && (
                <button
                  className="btn btn-ghost btn-xs"
                  onClick={async () => {
                    if (entry.documentId) {
                      // Pipeline failed — reingest existing document
                      try {
                        await reingestDocument(entry.documentId);
                        setEntries((prev) =>
                          prev.map((e, j) =>
                            j === i ? { ...e, status: "polling", error: undefined } : e,
                          ),
                        );
                      } catch (err) {
                        setEntries((prev) =>
                          prev.map((e, j) =>
                            j === i
                              ? { ...e, error: err instanceof Error ? err.message : "Retry failed" }
                              : e,
                          ),
                        );
                      }
                    } else {
                      // Upload itself failed — re-upload the file
                      setEntries((prev) => prev.filter((_, j) => j !== i));
                      void processFiles([entry.file]);
                    }
                  }}
                >
                  Retry
                </button>
              )}

              {entry.error && (
                <span className="text-xs text-muted" title={entry.error}>
                  ⚠
                </span>
              )}
            </div>
          ))}
        </div>
      )}

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

      {/* Existing documents for selected source */}
      {selectedSourceId && existingDocs.length > 0 && (() => {
        const uploadedIds = new Set(entries.map((e) => e.documentId).filter(Boolean));
        const filtered = existingDocs.filter((d) => !uploadedIds.has(d.id));
        if (filtered.length === 0) return null;
        return (
          <div style={{ marginTop: "2rem" }}>
            <h4 style={{ marginBottom: "0.5rem" }}>Source Documents</h4>
            <div className="file-list">
              {filtered.map((doc) => (
                <div key={doc.id} className="file-item">
                  <span className="file-item-name" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className="text-xs text-muted">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </span>

                  <StatusBadge status={doc.pipeline_status} />

                  {(doc.pipeline_status === "FAILED" || doc.pipeline_status === "ERROR" || doc.pipeline_status === "PENDING") && (
                    <button
                      className="btn btn-ghost btn-xs"
                      onClick={async () => {
                        try {
                          await reingestDocument(doc.id);
                          setExistingDocs((prev) =>
                            prev.map((d) =>
                              d.id === doc.id ? { ...d, pipeline_status: "PENDING" } : d,
                            ),
                          );
                        } catch (err) {
                          setError(err instanceof Error ? err.message : "Retry failed");
                        }
                      }}
                    >
                      Retry
                    </button>
                  )}

                  {doc.error_message && (
                    <span className="text-xs text-muted" title={doc.error_message}>
                      ⚠
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
