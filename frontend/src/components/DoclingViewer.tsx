import { useEffect, useState, useMemo } from "react";
import { getDoclingRawJson, getDoclingDocument, getDocumentMetadata, getDocumentImageDescriptions, getDocumentTranslation } from "../api/client";
import type { ImageDescription } from "../api/client";

interface DoclingViewerProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

type ViewMode = "document" | "translate" | "json";

function buildDoclingHtml(docJson: Record<string, unknown>): string {
  const jsonStr = JSON.stringify(docJson);
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="/static/docling-components.js" type="module"><\/script>
    <style>
      body { margin: 0; background: #f5f5f5; display: flex; justify-content: center; }
      docling-img { gap: 1rem; }
      docling-img::part(page) {
        box-shadow: 0 0.5rem 1rem 0 rgba(0, 0, 0, 0.2);
      }
    </style>
  </head>
  <body>
    <docling-img id="dclimg" pagenumbers>
      <docling-tooltip></docling-tooltip>
    </docling-img>

    <script id="dcljson" type="application/json">${jsonStr}<\/script>

    <script>
      (function() {
        function applySrc() {
          try {
            var data = JSON.parse(document.getElementById('dcljson').textContent);
            var el = document.getElementById('dclimg');
            if (el) el.src = data;
          } catch (e) {
            console.error('Failed to set docling-img src:', e);
          }
        }
        if (!customElements.get('docling-img')) {
          customElements.whenDefined('docling-img').then(applySrc);
        } else {
          applySrc();
        }
      })();
    <\/script>
  </body>
</html>`;
}

export function DoclingViewer({
  documentId,
  filename,
  onClose,
}: DoclingViewerProps) {
  const [docJson, setDocJson] = useState<Record<string, unknown> | null>(null);
  const [plainText, setPlainText] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [imageDescriptions, setImageDescriptions] = useState<ImageDescription[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("document");
  const [translatedMd, setTranslatedMd] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setDocJson(null);
    setPlainText(null);
    getDocumentMetadata(documentId).then(setMetadata);
    getDocumentImageDescriptions(documentId).then(setImageDescriptions);
    getDoclingRawJson(documentId)
      .then(setDocJson)
      .catch(() => {
        // Fallback: try fetching markdown/text content for non-PDF files
        return getDoclingDocument(documentId)
          .then((doc) => {
            if (doc.markdown) {
              setPlainText(doc.markdown);
            } else {
              setError("No viewable content available for this document.");
            }
          })
          .catch((err) =>
            setError(
              err instanceof Error ? err.message : "Failed to load document",
            ),
          );
      })
      .finally(() => setLoading(false));
  }, [documentId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const srcdoc = useMemo(() => {
    if (!docJson) return "";
    return buildDoclingHtml(docJson);
  }, [docJson]);

  return (
    <div className="docling-overlay" onClick={onClose}>
      <div className="docling-modal" onClick={(e) => e.stopPropagation()}>
        <div className="docling-modal-header">
          <h3 className="docling-modal-title" title={filename}>
            {filename}
          </h3>
          <div className="docling-mode-toggle">
            <button
              className={`mode-btn${mode === "document" ? " active" : ""}`}
              onClick={() => setMode("document")}
            >
              Document
            </button>
            {metadata && (metadata as any).has_translation && (
              <button
                className={`mode-btn${mode === "translate" ? " active" : ""}`}
                onClick={() => {
                  setMode("translate");
                  if (!translatedMd) {
                    getDocumentTranslation(documentId).then((t) => {
                      if (t) setTranslatedMd(t.translated_markdown);
                    });
                  }
                }}
              >
                Translate
              </button>
            )}
            {docJson && (
              <button
                className={`mode-btn${mode === "json" ? " active" : ""}`}
                onClick={() => setMode("json")}
              >
                JSON
              </button>
            )}
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="docling-modal-body">
          {loading && (
            <div className="empty-state">
              <span className="spinner" />
              <p className="mt-sm">Loading document...</p>
            </div>
          )}

          {error && <div className="alert alert-error">{error}</div>}

          {metadata && (
            <div style={{
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius)",
              padding: "0.75rem 1rem",
              marginBottom: "0.5rem",
              background: "var(--color-surface-2)",
              fontSize: "0.85rem",
            }}>
              <div style={{ fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-muted)" }}>
                AI-Extracted Document Metadata
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem 1.5rem" }}>
                <div><strong>Classification:</strong> {String(metadata.classification || "UNCLASSIFIED")}</div>
                <div><strong>Date of Information:</strong> {String(metadata.date_of_information || "Unknown")}</div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <strong>Source:</strong> {String(metadata.source_characterization || "Unknown")}
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <strong>Summary:</strong> {String(metadata.document_summary || "")}
                </div>
              </div>
            </div>
          )}

          {docJson && mode === "document" && (
            <iframe
              srcDoc={srcdoc}
              title={`DoclingViewer: ${filename}`}
              style={{
                width: "100%",
                height: "100%",
                border: "none",
                minHeight: "600px",
              }}
              sandbox="allow-scripts allow-same-origin"
            />
          )}

          {mode === "translate" && (
            <div style={{ padding: "1rem", overflowY: "auto", height: "100%" }}>
              <div style={{
                background: "#fff3cd",
                border: "1px solid #ffc107",
                borderRadius: "var(--radius, 4px)",
                padding: "0.5rem 0.75rem",
                marginBottom: "0.75rem",
                fontSize: "0.85rem",
              }}>
                Machine-translated from {(metadata as any)?.detected_language || "unknown language"}.
                Original may contain untranslated technical terms.
              </div>
              {translatedMd ? (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  fontFamily: "inherit",
                  fontSize: "0.9rem",
                  lineHeight: 1.6,
                  margin: 0,
                }}>
                  {translatedMd}
                </pre>
              ) : (
                <div className="empty-state">
                  <span className="spinner" />
                  <p className="mt-sm">Loading translation...</p>
                </div>
              )}
            </div>
          )}

          {/* Image description panel for standalone image files (no Docling JSON) */}
          {!docJson && imageDescriptions.length > 0 && mode === "document" && (
            <div style={{
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius)",
              padding: "0.75rem 1rem",
              marginBottom: "0.5rem",
              background: "var(--color-surface-2)",
              fontSize: "0.85rem",
            }}>
              <div style={{ fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-muted)" }}>
                AI Image Analysis
              </div>
              {imageDescriptions.map((desc) => (
                <div key={desc.element_uid} style={{ maxHeight: "300px", overflowY: "auto", whiteSpace: "pre-line", lineHeight: 1.5 }}>
                  {desc.content_text}
                </div>
              ))}
            </div>
          )}

          {/* Plain text fallback for .txt and files without Docling JSON */}
          {!docJson && plainText && mode === "document" && (
            <pre style={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: "0.9rem", lineHeight: 1.6, padding: "1rem", margin: 0 }}>
              {plainText}
            </pre>
          )}

          {docJson && mode === "json" && (
            <pre className="docling-json-content">
              {JSON.stringify(docJson, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
