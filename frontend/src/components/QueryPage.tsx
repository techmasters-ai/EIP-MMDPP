import React, { useState, useCallback } from "react";
import { unifiedQuery, getGraphNeighborhood, type QueryStrategy, type ModalityFilter, type QueryResultItem } from "../api/client";
import { GraphView, toGraphElements } from "./GraphView";
import type cytoscape from "cytoscape";

interface ModePreset {
  strategy: QueryStrategy;
  label: string;
  description: string;
}

const MODES: ModePreset[] = [
  {
    strategy: "basic",
    label: "Text Basic",
    description: "Simple BGE vector RAG search on text chunks",
  },
  {
    strategy: "hybrid",
    label: "Multi-Modal",
    description: "Hybrid pipeline (vectors + graph expansion)",
  },
  {
    strategy: "graphrag_local",
    label: "GraphRAG Local",
    description: "Entity-centric retrieval with community context reports",
  },
  {
    strategy: "graphrag_global",
    label: "GraphRAG Global",
    description: "Cross-community summarization for broad questions",
  },
  {
    strategy: "graphrag_drift",
    label: "GraphRAG Drift",
    description: "Community-informed expansion search (DRIFT)",
  },
  {
    strategy: "graphrag_basic",
    label: "GraphRAG Basic",
    description: "Vector search over GraphRAG text units",
  },
];

const MODALITY_OPTIONS: { value: ModalityFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "text", label: "Text Only" },
  { value: "image", label: "Images Only" },
];

function scoreColor(score: number): string {
  if (score >= 0.85) return "var(--color-success)";
  if (score >= 0.65) return "var(--color-primary)";
  return "var(--color-text-muted)";
}

function ImageLightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(0,0,0,0.8)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
      }}
      onClick={onClose}
    >
      <img
        src={src}
        alt={alt}
        style={{
          maxWidth: "90vw",
          maxHeight: "90vh",
          objectFit: "contain",
          borderRadius: "6px",
        }}
        onClick={(e) => e.stopPropagation()}
      />
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: "1rem",
          right: "1rem",
          background: "rgba(255,255,255,0.15)",
          border: "none",
          color: "#fff",
          fontSize: "1.5rem",
          cursor: "pointer",
          borderRadius: "50%",
          width: "2.5rem",
          height: "2.5rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        &times;
      </button>
    </div>
  );
}

/* ---------- GraphRAG Local: Entity explorer ---------- */
function GraphRAGLocalDetail({ ctx }: { ctx: Record<string, unknown> }) {
  const entity = ctx.entity as Record<string, unknown> | undefined;
  const reports = ctx.community_reports as Array<Record<string, unknown>> | undefined;

  return (
    <div style={{ marginTop: "0.5rem" }}>
      {entity && (
        <div style={{ marginBottom: "0.75rem" }}>
          <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
            Entity: {String(entity.name || "")}
            {entity.entity_type ? (
              <span className="badge badge-info" style={{ marginLeft: "0.5rem" }}>
                {String(entity.entity_type)}
              </span>
            ) : null}
          </div>
          <table style={{ fontSize: "0.85rem", borderCollapse: "collapse", width: "100%" }}>
            <tbody>
              {Object.entries(entity)
                .filter(([k]) => k !== "name" && k !== "entity_type")
                .map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
                    <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500, whiteSpace: "nowrap", verticalAlign: "top" }}>{k}</td>
                    <td style={{ padding: "0.2rem 0.5rem", wordBreak: "break-word" }}>
                      {typeof v === "object" ? JSON.stringify(v) : String(v ?? "")}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {reports && reports.length > 0 && (
        <div>
          <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Community Reports</div>
          {reports.map((r, i) => (
            <div key={i} style={{ marginBottom: "0.5rem", padding: "0.5rem", background: "var(--color-bg-muted, #f5f5f5)", borderRadius: "4px" }}>
              {r.title ? <div style={{ fontWeight: 500 }}>{String(r.title)}</div> : null}
              {r.summary ? <div className="text-sm" style={{ marginTop: "0.25rem" }}>{String(r.summary)}</div> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- GraphRAG Global: Community report explorer ---------- */
function GraphRAGGlobalDetail({ ctx }: { ctx: Record<string, unknown> }) {
  const [showFull, setShowFull] = useState(false);
  const reportText = ctx.report_text as string | undefined;

  return (
    <div style={{ marginTop: "0.5rem" }}>
      <div style={{ marginBottom: "0.25rem" }}>
        {ctx.community_title ? (
          <span style={{ fontWeight: 600 }}>{String(ctx.community_title)}</span>
        ) : null}
        {ctx.level != null && (
          <span className="text-xs text-muted" style={{ marginLeft: "0.5rem" }}>Level {String(ctx.level)}</span>
        )}
        {ctx.community_id ? (
          <span className="text-xs text-muted" style={{ marginLeft: "0.5rem" }}>ID: {String(ctx.community_id)}</span>
        ) : null}
      </div>
      {reportText && (
        <>
          <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>
            {showFull ? reportText : reportText.slice(0, 500) + (reportText.length > 500 ? "\u2026" : "")}
          </p>
          {reportText.length > 500 && (
            <button className="btn btn-ghost btn-sm" onClick={() => setShowFull((v) => !v)}>
              {showFull ? "Show less" : "Show full report"}
            </button>
          )}
        </>
      )}
    </div>
  );
}

/* ---------- Generic metadata detail section ---------- */
function MetadataDetail({ item }: { item: QueryResultItem }) {
  const ctx = item.context as Record<string, unknown> | undefined;

  return (
    <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <tbody>
          {item.chunk_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>chunk_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.chunk_id}</td>
            </tr>
          )}
          {item.artifact_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>artifact_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.artifact_id}</td>
            </tr>
          )}
          {item.document_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>document_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.document_id}</td>
            </tr>
          )}
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>score</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.score.toFixed(4)}</td>
          </tr>
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>modality</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.modality}</td>
          </tr>
          {item.page_number != null && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>page</td>
              <td style={{ padding: "0.2rem 0.5rem" }}>{item.page_number}</td>
            </tr>
          )}
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>classification</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.classification}</td>
          </tr>
          {ctx && Object.keys(ctx).length > 0 && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500, verticalAlign: "top" }}>context</td>
              <td style={{ padding: "0.2rem 0.5rem" }}>
                <pre style={{ margin: 0, fontSize: "0.8rem", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {JSON.stringify(ctx, null, 2)}
                </pre>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/** Extract entity names from GraphRAG context if graph data is present. */
function extractGraphEntities(ctx: Record<string, unknown> | undefined): string[] {
  if (!ctx) return [];
  const gctx = ctx.graphrag_context as Record<string, unknown> | undefined;
  if (!gctx) return [];

  // GraphRAG local/drift context includes entities as array of objects
  const entities = gctx.entities as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(entities) && entities.length > 0) {
    return entities
      .map((e) => (e.title as string) || (e.name as string) || "")
      .filter(Boolean);
  }

  // Fallback: check for reports with entity references
  const reports = gctx.reports as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(reports)) {
    const names: string[] = [];
    for (const r of reports) {
      const title = (r.title as string) || (r.entity_name as string);
      if (title) names.push(title);
    }
    if (names.length > 0) return names;
  }

  return [];
}

/* ---------- Result card ---------- */
function ResultCard({ item, index }: { item: QueryResultItem; index: number }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphElements, setGraphElements] = useState<cytoscape.ElementDefinition[] | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const displayText = item.content_text;
  const ctx = item.context as Record<string, unknown> | undefined;
  const isGraphRAGLocal = ctx?.source === "graphrag_local";
  const isGraphRAGGlobal = ctx?.source === "graphrag_global";
  const isGraphRAGDrift = ctx?.source === "graphrag_drift";
  const isGraphRAGBasic = ctx?.source === "graphrag_basic";

  // Show graph toggle only for GraphRAG results that contain entity/graph data
  const isGraphRAG = isGraphRAGLocal || isGraphRAGGlobal || isGraphRAGDrift || isGraphRAGBasic;
  const graphEntities = isGraphRAG ? extractGraphEntities(ctx) : [];
  const hasGraphData = graphEntities.length > 0;

  const handleToggleGraph = async () => {
    if (graphOpen) {
      setGraphOpen(false);
      setGraphElements(null);
      setGraphError(null);
      return;
    }
    setGraphOpen(true);
    setGraphLoading(true);
    setGraphError(null);
    try {
      // Fetch neighborhood for the first entity in the context
      const entityName = graphEntities[0];
      const resp = await getGraphNeighborhood({ entity_name: entityName, hop_count: 2 });
      const elements = toGraphElements(resp, entityName);
      if (elements.length === 0) {
        setGraphElements(null);
        setGraphError(`No graph data found for "${entityName}".`);
      } else {
        setGraphElements(elements);
      }
    } catch (err) {
      console.error("Graph neighborhood fetch failed:", err);
      setGraphElements(null);
      setGraphError(err instanceof Error ? err.message : "Failed to load graph data");
    } finally {
      setGraphLoading(false);
    }
  };

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
  } else if (isGraphRAGLocal) {
    const entityType = ctx.entity_type as string | undefined;
    provenanceLabel = `GraphRAG Local: ${entityType || "entity"} match`;
  } else if (isGraphRAGGlobal) {
    const title = ctx.community_title as string | undefined;
    provenanceLabel = `GraphRAG Global: ${title || "community report"}`;
  } else if (isGraphRAGDrift) {
    provenanceLabel = "GraphRAG Drift: community-informed expansion";
  } else if (isGraphRAGBasic) {
    provenanceLabel = "GraphRAG Basic: text unit vector search";
  }

  // Preview: first 300 chars always visible
  const previewLen = 300;
  const preview = displayText
    ? displayText.length > previewLen
      ? displayText.slice(0, previewLen) + "\u2026"
      : displayText
    : null;

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
        {hasGraphData && (
          <button
            className={`btn btn-ghost btn-sm graph-toggle-btn${graphOpen ? " active" : ""}`}
            onClick={() => void handleToggleGraph()}
            title="Toggle graph view"
          >
            ◉
          </button>
        )}
      </div>

      {provenanceLabel && (
        <div className="text-sm text-muted" style={{ marginBottom: "0.25rem" }}>
          {provenanceLabel}
        </div>
      )}

      {/* Inline image thumbnail */}
      {item.image_url && (
        <div style={{ margin: "0.5rem 0" }}>
          <img
            src={item.image_url}
            alt={item.content_text || "Retrieved image"}
            loading="lazy"
            style={{
              maxHeight: "200px",
              maxWidth: "100%",
              objectFit: "contain",
              borderRadius: "4px",
              border: "1px solid var(--color-border, #e0e0e0)",
              cursor: "pointer",
            }}
            title="Click for full size"
            onClick={() => setModalOpen(true)}
          />
          {modalOpen && (
            <ImageLightbox
              src={item.image_url}
              alt={item.content_text || "Retrieved image"}
              onClose={() => setModalOpen(false)}
            />
          )}
        </div>
      )}

      {/* Text preview — always visible */}
      {preview && (
        <p className="result-text">{preview}</p>
      )}

      {/* GraphRAG graph view (when toggled and entity data present) */}
      {graphOpen && (
        graphLoading ? (
          <div className="graph-view-container" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span className="spinner" />
          </div>
        ) : graphElements && graphElements.length > 0 ? (
          <GraphView
            elements={graphElements}
            onNodeClick={() => {}}
            onClose={() => { setGraphOpen(false); setGraphElements(null); setGraphError(null); }}
          />
        ) : (
          <div className="alert alert-error mt-sm">
            {graphError || "No graph data available."}
          </div>
        )
      )}

      {/* GraphRAG Local: entity details always shown inline */}
      {isGraphRAGLocal && ctx && <GraphRAGLocalDetail ctx={ctx} />}

      {/* GraphRAG Global: community report inline */}
      {isGraphRAGGlobal && ctx && <GraphRAGGlobalDetail ctx={ctx} />}

      {/* Show details toggle — full text + all metadata */}
      <button
        className="btn btn-ghost btn-sm mt-sm"
        onClick={() => setDetailsOpen((v) => !v)}
      >
        {detailsOpen ? "Hide details" : "Show details"}
      </button>

      {detailsOpen && (
        <div style={{ marginTop: "0.5rem", borderTop: "1px solid var(--color-border, #e0e0e0)", paddingTop: "0.5rem" }}>
          {/* Full text if longer than preview */}
          {displayText && displayText.length > previewLen && (
            <div style={{ marginBottom: "0.5rem" }}>
              <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Full Text</div>
              <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>{displayText}</p>
            </div>
          )}
          <MetadataDetail item={item} />
        </div>
      )}
    </div>
  );
}

export function QueryPage() {
  const [queryText, setQueryText] = useState("");
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [modalityFilter, setModalityFilter] = useState<ModalityFilter>("all");
  const [topK, setTopK] = useState(10);
  const [rerankerTopN, setRerankerTopN] = useState(20);
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [totalResults, setTotalResults] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const selected = MODES[selectedIdx];
  const showImageInput = selected.strategy === "hybrid";

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
        strategy: selected.strategy,
        modality_filter: selected.strategy === "hybrid" ? modalityFilter : "all",
        top_k: topK,
        reranker_top_n: rerankerTopN,
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
              {MODES.map((m, i) => (
                <button
                  key={m.strategy}
                  type="button"
                  className={`mode-btn${selectedIdx === i ? " active" : ""}`}
                  title={m.description}
                  onClick={() => {
                    setSelectedIdx(i);
                    setModalityFilter("all");
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {/* Modality sub-filter for Multi-Modal */}
            {selected.strategy === "hybrid" && (
              <div className="mode-selector" style={{ marginBottom: "1rem" }}>
                {MODALITY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    className={`mode-btn${modalityFilter === opt.value ? " active" : ""}`}
                    onClick={() => setModalityFilter(opt.value)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
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
            <div className="field" style={{ width: "110px", flexShrink: 0 }}>
              <label htmlFor="reranker-top-n">Reranker Top N</label>
              <input
                id="reranker-top-n"
                type="number"
                min={1}
                max={200}
                value={rerankerTopN}
                onChange={(e) => setRerankerTopN(parseInt(e.target.value, 10) || 20)}
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
        <code>GET /v1/agent/context?query=&hellip;&amp;strategy=basic&amp;top_k=10</code>
        <br />
        Returns a pre-formatted markdown context string ready for LLM prompt injection.
      </div>
    </div>
  );
}
