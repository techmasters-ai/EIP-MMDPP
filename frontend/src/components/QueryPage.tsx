import React, { useState, useCallback } from "react";
import { unifiedQuery, type QueryMode, type QueryResultItem } from "../api/client";

const MODES: { value: QueryMode; label: string; description: string }[] = [
  {
    value: "text_basic",
    label: "Text Basic",
    description: "Simple BGE vector RAG search on text chunks",
  },
  {
    value: "text_only",
    label: "Text Only",
    description: "Full multi-modal pipeline, filtered to text results",
  },
  {
    value: "images_only",
    label: "Images Only",
    description: "Full multi-modal pipeline, filtered to image results",
  },
  {
    value: "multi_modal",
    label: "Multi-Modal",
    description: "Full multi-modal pipeline, all results unfiltered",
  },
  {
    value: "memory",
    label: "Trusted Data",
    description: "Search approved trusted data",
  },
];

const IMAGE_MODES: Set<QueryMode> = new Set(["text_only", "images_only", "multi_modal"]);

function scoreColor(score: number): string {
  if (score >= 0.85) return "var(--color-success)";
  if (score >= 0.65) return "var(--color-primary)";
  return "var(--color-text-muted)";
}

function ResultCard({ item, index }: { item: QueryResultItem; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const displayText = item.content_text;
  const ctx = item.context as Record<string, unknown> | undefined;

  let provenanceLabel = "";
  if (ctx?.source === "ontology") {
    const entity = ctx.entity_name as string | undefined;
    const rel = ctx.rel_type as string | undefined;
    const related = ctx.related_name as string | undefined;
    if (entity && rel && related) {
      provenanceLabel = `Via ontology: ${entity} --[${rel}]--> ${related}`;
    }
  } else if (ctx?.source === "doc_structure") {
    const linkType = ctx.link_type as string | undefined;
    const hops = (ctx.hops as number) || 1;
    provenanceLabel = `Via document structure: ${linkType || "link"}${hops > 1 ? ` (${hops} hops)` : ""}`;
  } else if (ctx?.source === "cross_modal") {
    const edge = ctx.edge_type as string | undefined;
    if (edge) provenanceLabel = `Via graph bridge: ${edge}`;
  }

  const truncated = displayText && displayText.length > 400 && !expanded;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="text-xs text-muted">#{index + 1}</span>
        <span className="result-score" style={{ color: scoreColor(item.score) }}>
          {(item.score * 100).toFixed(0)}%
        </span>
        <span className="badge badge-info">{item.modality}</span>
        {item.classification && item.classification !== "UNCLASSIFIED" && (
          <span className="badge badge-error">{item.classification}</span>
        )}
        {item.page_number != null && (
          <span className="text-xs text-muted">p.{item.page_number}</span>
        )}
      </div>

      {provenanceLabel && (
        <div className="text-sm text-muted" style={{ marginBottom: "0.25rem" }}>
          {provenanceLabel}
        </div>
      )}

      {item.image_url && (
        <div className="result-image" style={{ margin: "0.5rem 0" }}>
          <img
            src={item.image_url}
            alt={item.content_text || "Retrieved image"}
            loading="lazy"
            style={{
              maxHeight: "300px",
              maxWidth: "100%",
              objectFit: "contain",
              borderRadius: "4px",
              border: "1px solid var(--color-border, #e0e0e0)",
            }}
          />
        </div>
      )}

      {displayText && (
        <>
          <p className="result-text">
            {truncated ? displayText.slice(0, 400) + "\u2026" : displayText}
          </p>
          {displayText.length > 400 && (
            <button
              className="btn btn-ghost btn-sm mt-sm"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </>
      )}

      <div className="result-meta">
        {item.chunk_id && <span>Chunk: {String(item.chunk_id).slice(0, 8)}&hellip;</span>}
        {item.artifact_id && <span>Artifact: {String(item.artifact_id).slice(0, 8)}&hellip;</span>}
      </div>
    </div>
  );
}

export function QueryPage() {
  const [queryText, setQueryText] = useState("");
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState<QueryMode>("text_basic");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [totalResults, setTotalResults] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const showImageInput = IMAGE_MODES.has(selectedMode);

  const handleImageFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImagePreview(dataUrl);
      setQueryImage(dataUrl.split(",")[1]);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleImageFile(file);
    },
    [handleImageFile],
  );

  const clearImage = () => {
    setQueryImage(null);
    setImagePreview(null);
  };

  const hasQuery = queryText.trim() || queryImage;

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hasQuery) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setTotalResults(0);
    setElapsed(null);
    const t0 = performance.now();

    try {
      const res = await unifiedQuery({
        query_text: queryText.trim() || undefined,
        query_image: queryImage || undefined,
        mode: selectedMode,
        top_k: topK,
        include_context: true,
      });
      setResults(res.results);
      setTotalResults(res.total);
      setElapsed(Math.round(performance.now() - t0));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleQuery(e)}>
          <div className="field">
            <label>Query mode</label>
            <div className="mode-selector" style={{ marginBottom: "1rem" }}>
              {MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  className={`mode-btn${selectedMode === m.value ? " active" : ""}`}
                  title={m.description}
                  onClick={() => setSelectedMode(m.value)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {showImageInput && (
            <div className="field" style={{ marginBottom: "1rem" }}>
              <label>Query image (optional &mdash; for image-based search)</label>
              {imagePreview ? (
                <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                  <img
                    src={imagePreview}
                    alt="Query preview"
                    style={{ maxWidth: "160px", maxHeight: "120px", borderRadius: "4px", border: "1px solid var(--color-border)" }}
                  />
                  <button type="button" className="btn btn-ghost btn-sm" onClick={clearImage}>
                    Remove
                  </button>
                </div>
              ) : (
                <div
                  className="drop-zone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById("query-image-input")?.click()}
                  style={{
                    cursor: "pointer",
                    textAlign: "center",
                    padding: "1.5rem",
                    border: "2px dashed var(--color-border)",
                    borderRadius: "6px",
                  }}
                >
                  <div className="text-muted text-sm">
                    Drop an image here or click to select &mdash; text-only queries also work for text-to-image search
                  </div>
                  <input
                    id="query-image-input"
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleImageFile(file);
                    }}
                  />
                </div>
              )}
            </div>
          )}

          <div className="field-row" style={{ alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="query-input">Search query</label>
              <input
                id="query-input"
                type="search"
                placeholder="e.g. Patriot PAC-3 guidance computer specifications"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                autoFocus
              />
            </div>
            <div className="field" style={{ width: "80px", flexShrink: 0 }}>
              <label htmlFor="top-k">Top K</label>
              <input
                id="top-k"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
              />
            </div>
            <div style={{ paddingBottom: "0" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !hasQuery}
              >
                {loading ? <><span className="spinner" /> Searching&hellip;</> : "Search"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && <div className="alert alert-error" style={{ marginTop: "1rem" }}>{error}</div>}

      {results !== null && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {totalResults === 0
                ? "No results found."
                : `${totalResults} result${totalResults !== 1 ? "s" : ""}`}
              {elapsed != null && ` \u2014 ${elapsed}ms`}
            </span>
          </div>

          {totalResults === 0 ? (
            <div className="empty-state">
              <div className="empty-state-title">No matching documents</div>
              <div className="text-muted text-sm">
                Try a different query or upload documents first.
              </div>
            </div>
          ) : (
            results.map((item: QueryResultItem, i: number) => (
              <ResultCard key={`result-${i}`} item={item} index={i} />
            ))
          )}
        </div>
      )}

      <div className="api-info mt-md">
        <strong>LangGraph tool endpoint:</strong>{" "}
        <code>GET /v1/agent/context?query=&hellip;&amp;mode=text_basic&amp;top_k=10</code>
        <br />
        Returns a pre-formatted markdown context string ready for LLM prompt injection.
      </div>
    </div>
  );
}
