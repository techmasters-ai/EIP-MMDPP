import { useEffect, useState, useMemo } from "react";
import { getDoclingRawJson } from "../api/client";

interface DoclingViewerProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

type ViewMode = "document" | "json";

function buildDoclingHtml(docJson: Record<string, unknown>): string {
  const jsonStr = JSON.stringify(docJson);
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="/static/docling-components.js" type="module"><\/script>
    <style>
      body { margin: 0; background: #f5f5f5; }
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("document");

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDoclingRawJson(documentId)
      .then(setDocJson)
      .catch((err) =>
        setError(
          err instanceof Error ? err.message : "Failed to load document",
        ),
      )
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
