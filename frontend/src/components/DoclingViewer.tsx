import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getDoclingDocument, type DoclingDocumentResponse } from "../api/client";

interface DoclingViewerProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

type ViewMode = "markdown" | "json";

export function DoclingViewer({ documentId, filename, onClose }: DoclingViewerProps) {
  const [data, setData] = useState<DoclingDocumentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("markdown");

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDoclingDocument(documentId)
      .then(setData)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load document"),
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

  return (
    <div className="docling-overlay" onClick={onClose}>
      <div className="docling-modal" onClick={(e) => e.stopPropagation()}>
        <div className="docling-modal-header">
          <h3 className="docling-modal-title" title={filename}>
            {filename}
          </h3>
          <div className="docling-mode-toggle">
            <button
              className={`mode-btn${mode === "markdown" ? " active" : ""}`}
              onClick={() => setMode("markdown")}
            >
              Document
            </button>
            <button
              className={`mode-btn${mode === "json" ? " active" : ""}`}
              onClick={() => setMode("json")}
            >
              JSON
            </button>
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

          {data && mode === "markdown" && (
            <div className="docling-markdown-content">
              <Markdown
                remarkPlugins={[remarkGfm]}
                components={{
                  img: ({ src, alt, ...props }) => {
                    const matchedImage = data.images.find(
                      (img) => src?.includes(img.element_uid),
                    );
                    const resolvedSrc = matchedImage
                      ? matchedImage.url
                      : src?.startsWith("/") || src?.startsWith("http")
                        ? src
                        : undefined;
                    return (
                      <img
                        src={resolvedSrc}
                        alt={alt || ""}
                        className="docling-inline-image"
                        {...props}
                      />
                    );
                  },
                }}
              >
                {data.markdown}
              </Markdown>
              {/* Show images not embedded in markdown */}
              {(() => {
                const markdownText = data.markdown || "";
                const unmatched = data.images.filter(
                  (img) => !markdownText.includes(img.element_uid),
                );
                return unmatched.length > 0 ? (
                  <div className="docling-image-gallery">
                    {unmatched.map((img) => (
                      <img
                        key={img.element_uid}
                        src={img.url}
                        alt={img.element_uid}
                        className="docling-inline-image"
                      />
                    ))}
                  </div>
                ) : null;
              })()}
            </div>
          )}

          {data && mode === "json" && (
            <pre className="docling-json-content">
              {JSON.stringify(data.document_json, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
